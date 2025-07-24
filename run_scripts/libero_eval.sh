#!/bin/bash

#SBATCH --partition="kira-lab,overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH --mem-per-gpu=45G

export HOME="/coc/testnvme/chuang475"
export CONDA_BASE_PATH="${HOME}/miniconda3"
export CONDA_ENV_NAME="openvla" # openvla rlds_env
export PYTHON_BIN="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/python"
export PRISMATIC_DATA_ROOT="/coc/testnvme/chuang475/datasets"
export HUGGINGFACE_HUB_CACHE="/coc/testnvme/chuang475/huggingface_cache"

# cd /coc/testnvme/chuang475/projects/openvla-mini

# Launch LIBERO-90 evals
# srun -u ${PYTHON_BIN} -m experiments.robot.libero.run_libero_eval \
#   --model_family openvla \
#   --pretrained_checkpoint openvla/openvla-7b \
#   --task_suite_name libero_90 \
#   --center_crop True \

# srun -u ${PYTHON_BIN} -m experiments.robot.libero.run_libero_eval \
#   --model_family prismatic \
#   --pretrained_checkpoint /coc/testnvme/chuang475/projects/openvla-mini/log/prism-qwen25-dinosiglip-224px+0_5b+mx-libero-90+n1+b24+lr2e-05+e10+x7/checkpoints/step-195000-epoch-55-loss=0.0254.pt \
#   --task_suite_name libero_90 \
#   --center_crop True \

srun -u ${PYTHON_BIN} -m experiments.robot.libero.run_libero_eval \
  --model_family prismatic \
  --pretrained_checkpoint /coc/testnvme/chuang475/projects/openvla-mini/log/minivla-libero90-prismatic/checkpoints/step-122500-epoch-55-loss=0.0743.pt \
  --task_suite_name libero_90 \
  --center_crop True \

# export CUDA_VISIBLE_DEVICES= # disables GPU usage
# cd /coc/testnvme/chuang475/projects/rlds_dataset_builder/rlbench_rlds_convertor/LIBERO_90

# tfds build --overwrite