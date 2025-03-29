export EXPERIMENT_DIR="/experiments/isic-stable-diffusion"
export DATA_PATH="$DATA_ROOT/isic_mel_balanced/"  # Path to the isic data directory
export SEED=42

# Data parameters
export IMAGE_SIZE=256                   # (int) Size of the input images          
export IMAGE_CHANNELS=3                 # (int) Number of channels in the input images

export BATCH_SIZE=128                     # (int) Batch size for training
export EVALUATION_BATCHES=80            # (int) Number of batches to evaluate on
export MIXED_PRECISION="fp16"           # (str) Mixed precision training ('fp16' or 'fp32' or 'none')
export NUM_WORKERS=24                   # (int) Number of workers for the data loader

# Classification parameters
export CLASSES=2                        # (int) Number of classes in the dataset
export CLASSIFICATION=true              # (bool) Whether to perform classification or not
export MAJORITY_VOTING=false            # (bool) Whether to perform majority voting or not
export N_STAGES=1                       # (int) Number of stages for the classification
export EVALUATION_PER_STAGE=[51]       # (list) Number of samples to evaluate per stage
export N_KEEP_PER_STAGE=[1]             # (list) Number of classes to keep per stage (Must end with 1)

# Model parameters
export VERSION="2-0"
export MODEL_PATH="$INFERENCE_CHECKPOINT_FOLDER/stable-diffusion/"       # (str) Path to the stable diffusion model

export CONFIG="{
  \"project_root\": \"$PROJECT_ROOT\",
  \"experiment_dir\": \"$EXPERIMENT_DIR\",
  \"data_path\": \"$DATA_PATH\",
  \"image_size\": $IMAGE_SIZE,
  \"image_channels\": $IMAGE_CHANNELS,
  \"batch_size\": $BATCH_SIZE,
  \"evaluation_batches\": $EVALUATION_BATCHES,
  \"mixed_precision\": \"$MIXED_PRECISION\",
  \"num_workers\": $NUM_WORKERS,
  \"classes\": $CLASSES,
  \"seed\": $SEED,
  \"classification\": $CLASSIFICATION,
  \"n_stages\": $N_STAGES,
  \"evaluation_per_stage\": $EVALUATION_PER_STAGE,
  \"n_keep_per_stage\": $N_KEEP_PER_STAGE,
  \"majority_voting\": $MAJORITY_VOTING,
  \"version\": \"$VERSION\",
  \"model_path\": \"$MODEL_PATH\"
}"

port=$(shuf -i 1025-65535 -n 1)
accelerate launch \
                  --main-process-port=$port \
                  --num-machines=1 \
                  --num-processes=1 \
                  --mixed_precision='fp16' \
                  $PROJECT_ROOT$EXPERIMENT_DIR/inference.py