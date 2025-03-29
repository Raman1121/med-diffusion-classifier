###### Shared parameters for training/inference/explain ######
export EXPERIMENT_DIR="/experiments/chexpert-classifier"
export DATA_PATH="$DATA_ROOT/chexpert"  # (str) Path to the data directory containing the dataset
export SEED=42

# Data parameters
export IMAGE_SIZE=256                   # (int) Size of the input images, 224 for vit/swin and 256 for resnet/efficientnet          
export IMAGE_CHANNELS=3                 # (int) Number of channels in the input images

# Optimizer parameters
export BATCH_SIZE=64                     # (int) Batch size for training
export NUM_EPOCHS=100                   # (int) Number of epochs to train for
export GRADIENT_ACCUMULATION_STEPS=1    # (int) Number of gradient accumulation steps
export LEARNING_RATE=0.0001             # (float) Learning rate
export WEIGHT_DECAY=0.001
export LR_WARMUP_STEPS=20              # (int) Number of warmup steps for the learning rate
export EVALUATION_BATCHES=80            # (int) Number of batches to evaluate on
export EVAL_PERIOD=1                    # (int) Number of epochs between evaluation
export MIXED_PRECISION="fp16"           # (str) Mixed precision training ('fp16' or 'fp32' or 'none')
export NUM_WORKERS=24                   # (int) Number of workers for the data loader

# Model parameters
export CLASSES=2                        # (int) Number of classes in the dataset
export PRETRAINED=true                  # (bool) Whether to use a pretrained model or not

###### Training parameters ######
export RESUME=0                 # (int) Resume training from the last checkpoint

###### Inference parameters ######
export CHECKPOINT_FOLDER="$INFERENCE_CHECKPOINT_FOLDER/chexpert-classifier/best_checkpoint_$VARIANT"           # (str) Checkpoint folder: empty if loading default based on variant

export CONFIG="{
  \"resume\": $RESUME,
  \"project_root\": \"$PROJECT_ROOT\",
  \"experiment_dir\": \"$EXPERIMENT_DIR\",
  \"data_path\": \"$DATA_PATH\",
  \"image_size\": $IMAGE_SIZE,
  \"image_channels\": $IMAGE_CHANNELS,
  \"batch_size\": $BATCH_SIZE,
  \"num_epochs\": $NUM_EPOCHS,
  \"gradient_accumulation_steps\": $GRADIENT_ACCUMULATION_STEPS,
  \"learning_rate\": $LEARNING_RATE,
  \"weight_decay\": $WEIGHT_DECAY,
  \"lr_warmup_steps\": $LR_WARMUP_STEPS,
  \"evaluation_batches\": $EVALUATION_BATCHES,
  \"mixed_precision\": \"$MIXED_PRECISION\",
  \"num_workers\": $NUM_WORKERS,
  \"classes\": $CLASSES,
  \"seed\": $SEED,
  \"use_comet\": $USE_COMET,
  \"comet_project_name\": \"$COMET_PROJECT_NAME\",
  \"comet_workspace\": \"$COMET_WORKSPACE\",
  \"comet_experiment_name\": \"$COMET_EXPERIMENT_NAME\",
  \"comet_api_key\": \"$COMET_API_KEY\",
  \"variant\": \"$VARIANT\",
  \"eval_period\": $EVAL_PERIOD,
  \"pretrained\": $PRETRAINED,
  \"backbone\": \"$BACKBONE\",
  \"checkpoint_folder\": \"$CHECKPOINT_FOLDER\"
}"

# Run the Python script
port=$(shuf -i 1025-65535 -n 1)
accelerate launch \
                  --main-process-port=$port \
                  --num-machines=1 \
                  --num-processes=1 \
                  --mixed_precision='fp16' \
                  $PROJECT_ROOT$EXPERIMENT_DIR/$FUNCTION.py # train.py or inference.py