export MODEL_NAME="stabilityai/stable-diffusion-2-base"
export TRAIN_DIR="$DATA_ROOT/sd_isic_chexpert"
export EXPERIMENT_DIR="/experiments/stable-diffusion-medical"
export OUTPUT_DIR="$PROJECT_ROOT/sd_chexpert_finetuning_output"

accelerate launch \
  --num-machines=1 \
  --num-processes=1 \
  --gpu_ids=0 \
  --mixed_precision="fp16" \
  $PROJECT_ROOT$EXPERIMENT_DIR/train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=25000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_DIR \
  --checkpointing_steps=100 \
  --resume_from_checkpoint=latest \
  --checkpoints_total_limit=3
