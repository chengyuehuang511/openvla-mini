"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer, LanguageActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        save_every_n_steps: Optional[int] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.llm_backbone.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # how often to save checkpoints
        self.save_every_n_steps = save_every_n_steps
        if save_every_n_steps is not None:
            assert save_every_n_steps > 0

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    def run_training(
        self,
        dataset: Dataset,
        collator: PaddedCollatorForLanguageModeling,
        metrics: Metrics,
        stage: str = "finetune",
        batch_construction_strategy: str = "split-modality",
        seed: int = 7,
    ) -> None:
        """Run the training loop for the given `dataset` and `collator`; log losses, results to `metrics`"""
        if "finetune" in stage and batch_construction_strategy == "split-modality":
            # Instantiate the split-modality sampler; if you want to extend with other batch construction schemes,
            #   (e.g., grouping by length) =>> can easily add them here!
            modality_lengths = dataset.get_modality_lengths()
            sampler = SplitModalitySampler(
                dataset,
                modality_lengths,
                global_batch_size=self.global_batch_size,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                seed=seed,
                drop_last=False,
            )

        else:
            sampler = DistributedSampler(
                dataset,
                num_replicas=overwatch.world_size(),
                rank=overwatch.rank(),
                shuffle=True,
                seed=seed,
                drop_last=False,
            )

        # Create a DataLoader with the initialized sampler, per-device-bsz, and collator
        dataloader = DataLoader(
            dataset,
            batch_size=self.per_device_batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            worker_init_fn=self.worker_init_fn,
        )

        # Max Steps vs. Epochs Computation
        steps_per_epoch = len(dataloader) // self.grad_accumulation_steps
        if self.max_steps is not None and steps_per_epoch < self.max_steps:
            # Just set `epochs` to some large number --> we'll short-circuit based on steps anyway
            self.epochs = 100

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(
                (self.epochs * (len(dataloader) // self.grad_accumulation_steps))
                if self.max_steps is None
                else self.max_steps
            ),
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            for epoch in range(self.epochs):
                self.vlm.train()
                sampler.set_epoch(epoch)

                # Zero-Gradients (just in case)
                self.optimizer.zero_grad()

                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                for train_idx, batch in enumerate(dataloader):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    with torch.autocast(
                        "cuda",
                        dtype=self.mixed_precision_dtype,
                        enabled=self.enable_mixed_precision_training,
                    ):
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            multimodal_indices=batch["multimodal_indices"],
                        )
                        loss = output.loss

                    # Commit Loss (Prior to Gradient Accumulation Normalization)
                    metrics.commit(loss=loss)

                    # Normalize Loss to account for Gradient Accumulation --> Backward!
                    # [IMPORTANT] Technically speaking, doing gradient accumulation in this way is "incorrect"; this is
                    #             because in general, each batch has a *different number of masked out tokens* (because
                    #             we're instruct-tuning). Taking the mean over two unbalanced means != the right thing!
                    #
                    #             HOWEVER -- at least at the 7B scale, the "naive" approach is just as performant as
                    #             the "correct" implementation, without adding extra complexity.
                    #
                    # That being said =>> at the 13B scale, *no matter what we tried, ANY gradient accumulation is just
                    #   really bad for downstream performance. Initial investigation shows that BF16 accumulation
                    #   just really tanks in precision... and don't have a good/clean way to fix this. Would love for
                    #   someone to PR and fix this (and I'd greatly appreciate it!!!)
                    normalized_loss = loss / self.grad_accumulation_steps
                    normalized_loss.backward()

                    # Step =>> Only if Done w/ Gradient Accumulation
                    if (train_idx + 1) % self.grad_accumulation_steps == 0:
                        metrics.commit(update_step_time=True)

                        # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality-assumptions
                        self.clip_grad_norm()

                        # Optimizer & LR Scheduler Step
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()

                        # Push Metrics
                        metrics.commit(global_step=metrics.global_step + 1, lr=self.lr_scheduler.get_last_lr()[0])
                        status = metrics.push()

                        # Check for Termination & Save Final Checkpoint (in case `max_steps` is not None)
                        if self.max_steps is not None and metrics.global_step >= self.max_steps:
                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                            return
                        elif (
                            self.save_every_n_steps is not None
                            and (metrics.global_step + 1) % self.save_every_n_steps == 0
                        ):

                            self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                            dist.barrier()

                        # Update Progress Bar
                        progress.update()
                        progress.set_description(status)

            # Save checkpoint at end each epoch (if `self.max_steps` is None)
            if self.max_steps is None:
                self.save_checkpoint(metrics.run_dir, metrics.global_step, epoch, loss.item())
                dist.barrier()

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch in dataloader:
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                with torch.autocast(
                    "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                ):
                    # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                    output: CausalLMOutputWithPast = self.vlm(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        pixel_values=batch["pixel_values"],
                        labels=batch["labels"],
                    )
                    loss = output.loss
                    batch_type = ["vqa" if "vqa" in name.decode() else "action" for name in batch["dataset_names"]]
                    is_action = torch.tensor([bt == "action" for bt in batch_type], device=output.logits.device)
                    shift_logits = output.logits[:, self.vlm.vision_backbone.num_patches : -1]
                    shift_labels = batch["labels"][:, 1:].to(shift_logits.device)

                    if isinstance(action_tokenizer, LanguageActionTokenizer):
                        # Sanity Check Loss
                        sanity_check_loss = torch.nn.functional.cross_entropy(
                            shift_logits.reshape(-1, shift_logits.size(-1)),  # [B * T, V]
                            shift_labels.reshape(-1),                         # [B * T]
                        )
                        assert torch.isclose(
                            loss, sanity_check_loss, atol=1e-1
                        ), f"Loss mismatch: {loss.item()} != {sanity_check_loss.item()}"
                        
                        # Initialize logits mask: [B, T, V]
                        mask = torch.full_like(shift_logits, -1e8)  # shape: (B, T, V)

                        # Apply token-specific masking per sample
                        for i, bt in enumerate(batch_type):
                            # assert action_tokenizer.tokenizer.vocab_size == action_tokenizer.action_token_begin_idx + 1, \
                            #     f"Action tokenizer vocab size ({action_tokenizer.tokenizer.vocab_size}) must match action token begin index + 1 ({action_tokenizer.action_token_begin_idx + 1})!"

                            if bt == "action":
                                # Allow only action token range
                                # TODO: sanity check the mask index
                                mask[i, :, action_tokenizer.tokenizer.vocab_size:] = 0
                            elif bt == "vqa":
                                # Allow only language token range
                                mask[i, :, :action_tokenizer.action_token_begin_idx] = 0 # action_tokenizer.tokenizer.vocab_size
                            else:
                                raise ValueError(f"Unsupported `batch_type` = {bt}")

                        # Apply Mask to Shifted Logits and update Loss
                        shift_logits += mask
                        loss = torch.nn.functional.cross_entropy(
                            shift_logits.reshape(-1, shift_logits.size(-1)),  # [B * T, V]
                            shift_labels.reshape(-1),                         # [B * T]
                        )

                # Commit Loss =>> Backward!
                metrics.commit(loss=loss)
                loss.backward()

                # === Compute Action Token Accuracy & L1 Loss ===

                # To compute action token accuracy, we need to identify the locations of the action tokens
                # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                # insert `self.vlm.vision_backbone.num_patches` at index 1.
                #
                # Computing `action_prediction_accuracy` is then pretty straightforward:
                #   1) Extract "aligned" predictions & labels
                #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                action_preds = shift_logits.argmax(dim=2)
                action_gt = shift_labels.to(action_preds.device)
                valid_token_mask = (action_tokenizer.action_token_end_idx > action_gt) & (action_gt > action_tokenizer.action_token_begin_idx)

                # === Mask for action samples only ===
                # Expand to match (B, T, V) shape
                sample_mask = is_action.unsqueeze(1).expand(-1, valid_token_mask.shape[1])
                # Final mask: valid action tokens from action-type samples only
                mask = valid_token_mask & sample_mask  # (B, T, V)

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float().clamp(min=1)

                # # Compute L1 Loss on Predicted (Continuous) Actions
                # continuous_actions_pred = torch.tensor(
                #     action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
                # )
                # continuous_actions_gt = torch.tensor(
                #     action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
                # )
                # action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

                # === L1 Loss ===
                # Use only masked entries
                pred_ids = action_preds[mask]  # (N,)
                gt_ids = action_gt[mask]       # (N,)

                # Decode to continuous action values
                if pred_ids.numel() > 0:
                    continuous_actions_pred = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(pred_ids.cpu().numpy())
                    )
                    continuous_actions_gt = torch.tensor(
                        action_tokenizer.decode_token_ids_to_actions(gt_ids.cpu().numpy())
                    )
                    action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
                else:
                    action_l1_loss = torch.tensor(0.0)

                # Commit Metrics
                metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                if overwatch.is_rank_zero():
                    datasets = set(batch["dataset_names"])
                    if len(datasets) > 1:
                        for ds in datasets:
                            ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                            action_accuracy_ds = correct_preds[ds_mask].sum().float() / mask[ds_mask].sum().float()
                            continuous_actions_pred_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_preds[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            continuous_actions_gt_ds = torch.tensor(
                                action_tokenizer.decode_token_ids_to_actions(
                                    action_gt[ds_mask][mask[ds_mask]].cpu().numpy()
                                )
                            )
                            action_l1_loss_ds = torch.nn.functional.l1_loss(
                                continuous_actions_pred_ds, continuous_actions_gt_ds
                            )
                            metrics.commit_for_dataset(
                                dataset_name=ds.decode(), action_accuracy=action_accuracy_ds, l1_loss=action_l1_loss_ds
                            )

                # === Gradient Step ===

                # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                self.clip_grad_norm()

                # Optimizer & LR Scheduler Step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Compute epoch value using number of completed gradient steps
                epoch = (metrics.global_step + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=metrics.global_step + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and metrics.global_step >= self.max_steps)) or (
                    (metrics.global_step % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, metrics.global_step, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    # # Merge LoRA weights into model backbone for faster inference
                    # #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    # if cfg.use_lora:
                    #     base_vla = AutoModelForVision2Seq.from_pretrained(
                    #         cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    #     )
                    #     merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    #     merged_vla = merged_vla.merge_and_unload()
                    #     if distributed_state.is_main_process:
                    #         if cfg.save_latest_checkpoint_only:
                    #             # Overwrite latest checkpoint
                    #             merged_vla.save_pretrained(run_dir)

                    #             print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                    #         else:
                    #             # Prepare to save checkpoint in new directory
                    #             checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                    #             os.makedirs(checkpoint_dir, exist_ok=True)

                    #             # Save dataset statistics to new directory
                    #             save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                    #             # Save processor and model weights to new directory
                    #             processor.save_pretrained(checkpoint_dir)
                    #             merged_vla.save_pretrained(checkpoint_dir)

                    #             print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")
                    
                    # dist.barrier()

                    if terminate:
                        return

                # Update Progress Bar
                progress.update()
                progress.set_description(status)
