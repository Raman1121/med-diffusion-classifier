###### Shared parameters for training/inference/explain ######
export EXPERIMENT_DIR="/experiments/isic-unet"
export DATA_PATH="$DATA_ROOT/isic_mel_balanced"   # Path to the isic data directory
export SEED=42

# Data parameters
export IMAGE_SIZE=256                   # (int) Size of the input images          
export IMAGE_CHANNELS=3                 # (int) Number of channels in the input images
export WAVELET_TRANSFORM=true           # (bool) Whether to use the wavelet transform or not

# Optimizer/EMA parameters
export BATCH_SIZE=128                    # (int) Batch size for training
export NUM_EPOCHS=700                  # (int) Number of epochs to train for
export GRADIENT_ACCUMULATION_STEPS=1    # (int) Number of gradient accumulation steps
export LEARNING_RATE=0.0001             # (float) Learning rate
export LR_WARMUP_STEPS=100              # (int) Number of warmup steps for the learning rate
export EVALUATION_BATCHES=80             # (int) Number of batches to evaluate on
export MIXED_PRECISION="fp16"           # (str) Mixed precision training ('fp16' or 'fp32' or 'none')
export NUM_WORKERS=6                    # (int) Number of workers for the data loader
export SAVE_IMAGE_EPOCHS=50             # (int) Number of epochs between saving images/evaluation

export EMA_BETA=0.999                   # (float) Exponential moving average beta
export EMA_WARMUP=50                    # (int) Number of warmup steps for the exponential moving average
export EMA_UPDATE_FREQ=5                # (int) Number of steps between EMA updates

# Model parameteres
export PRED_PARAM="v"                   # (str) Diffusion parameterization ('v' or 'eps')
export SCHEDULE="shifted_cosine"        # (str) Learning rate schedule ('cosine', 'shifted_cosine')
export NOISE_D=64                       # (int) Reference noise dimensionality (simple diffusion, Hoogeboom et al. 2023)
export ENCODER_TYPE="nn"                # (str) Type of encoder for the end-to-end model ('nn' or 't5')

# Classification parameters
export CLASSES=2                        # (int) Number of classes in the dataset
export CLASSIFICATION=true             # (bool) Whether to perform classification or not
export N_STAGES=1                       # (int) Number of stages for the classification
export EVALUATION_PER_STAGE=[51]        # (list) Number of samples to evaluate per stage
export N_KEEP_PER_STAGE=[1]             # (list) Number of classes to keep per stage (Must end with 1)
export MAJORITY_VOTING=true            # (bool) Whether to perform majority voting or not

###### Training parameters ######
export RESUME=0

###### Inference/Explain parameters ######
export CHECKPOINT_FOLDER="$INFERENCE_CHECKPOINT_FOLDER/isic-256-checkpoints/isic-unet-dwt"           # (str) Checkpoint folder for inference 
export FLASH_ATTENTION=false            # (bool) Whether to use the flash attention or not

export CFG_W=4.5                        # (int) Classifier guidance scale
export SAMPLING_STEPS=50               # (int) Number of sampling steps for the reverse diffusion process
export FROM_T=0.5

export CONFIG="{
  \"resume\": $RESUME,
  \"checkpoint_folder\": \"$CHECKPOINT_FOLDER\",
  \"project_root\": \"$PROJECT_ROOT\",
  \"experiment_dir\": \"$EXPERIMENT_DIR\",
  \"data_path\": \"$DATA_PATH\",
  \"image_size\": $IMAGE_SIZE,
  \"wavelet_transform\": $WAVELET_TRANSFORM,
  \"image_channels\": $IMAGE_CHANNELS,
  \"batch_size\": $BATCH_SIZE,
  \"num_epochs\": $NUM_EPOCHS,
  \"gradient_accumulation_steps\": $GRADIENT_ACCUMULATION_STEPS,
  \"learning_rate\": $LEARNING_RATE,
  \"lr_warmup_steps\": $LR_WARMUP_STEPS,
  \"save_image_epochs\": $SAVE_IMAGE_EPOCHS,
  \"evaluation_batches\": $EVALUATION_BATCHES,
  \"ema_beta\": $EMA_BETA,
  \"ema_warmup\": $EMA_WARMUP,
  \"ema_update_freq\": $EMA_UPDATE_FREQ,
  \"pred_param\": \"$PRED_PARAM\",
  \"schedule\": \"$SCHEDULE\",
  \"noise_d\": $NOISE_D,
  \"mixed_precision\": \"$MIXED_PRECISION\",
  \"num_workers\": $NUM_WORKERS,
  \"classes\": $CLASSES,
  \"encoder_type\": \"$ENCODER_TYPE\",
  \"cfg_w\": $CFG_W,
  \"sampling_steps\": $SAMPLING_STEPS,
  \"seed\": $SEED,
  \"use_comet\": $USE_COMET,
  \"comet_project_name\": \"$COMET_PROJECT_NAME\",
  \"comet_workspace\": \"$COMET_WORKSPACE\",
  \"comet_experiment_name\": \"$COMET_EXPERIMENT_NAME\",
  \"comet_api_key\": \"$COMET_API_KEY\",
  \"classification\": $CLASSIFICATION,
  \"n_stages\": $N_STAGES,
  \"evaluation_per_stage\": $EVALUATION_PER_STAGE,
  \"n_keep_per_stage\": $N_KEEP_PER_STAGE,
  \"majority_voting\": $MAJORITY_VOTING,
  \"flash_attention\": $FLASH_ATTENTION,
  \"from_t\": $FROM_T
}"

port=$(shuf -i 1025-65535 -n 1)
accelerate launch \
                --main-process-port=$port \
                --num-machines=1 \
                --num-processes=1 \
                --mixed_precision='fp16' \
                $PROJECT_ROOT$EXPERIMENT_DIR/$FUNCTION.py   # train.py or inference.py or explain.py
