#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="long"
#SBATCH --mem-per-gpu=45G

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="openvla-new"
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"
export HUGGINGFACE_HUB_CACHE="/coc/testnvme/chuang475/huggingface_cache"

cd /coc/testnvme/chuang475/projects/openvla-mini

# srun -u ${PYTHON_BIN} -m torch.distributed.run \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=8 \
#   vla-scripts/train.py \
#   --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-10-no_noops" \
#   --data_root_dir "/coc/testnvme/chuang475/projects/openvla-mini/data/modified_libero_rlds" \
#   --run_root_dir "log/" \
#   --wandb_project "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-10-no_noops" \
#   --wandb_entity "chuang475-georgia-institute-of-technology" \

srun -u ${PYTHON_BIN} -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=8 \
  vla-scripts/train.py \
  --vla.type "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90-no_noops-vqav2" \
  --data_root_dir "/coc/testnvme/chuang475/projects/openvla-mini/data/modified_libero_rlds" \
  --run_root_dir "log/" \
  --wandb_project "prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90-no_noops-vqav2" \
  --wandb_entity "chuang475-georgia-institute-of-technology" \
  # --pretrained_checkpoint "log/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90-no_noops+n1+b24+lr2e-05+e10+x7/checkpoints/step-110000-epoch-36-loss=0.0409.pt" \
  # --resume_step 110000 \
  # --resume_epoch 36 \

  #"prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90" \