export PROJECT_ROOT="/raid/s2198939/med-diffusion-classifier"   # Path to diffusion-classifier repository
export DATA_ROOT="/raid/s2198939/med-diffusion-classifier/data" # Path to the data directory containing chexpert and mel_isic_balanced  
export INFERENCE_CHECKPOINT_FOLDER="/raid/s2198939/med-diffusion-classifier/final-models" # Checkpoint folder for inference
# export INFERENCE_CHECKPOINT_FOLDER="/raid/s2198939/med-diffusion-classifier/sd_chexpert_finetuning_output"

export COMET_PROJECT_NAME="diffusion-classifier"  
export COMET_WORKSPACE="diffusion-classifier"
export COMET_API_KEY=""
export COMET_EXPERIMENT_NAME="diffusion-classifier-fairness"         
export USE_COMET=1

export MODEL="sd"                           # "baseline", "unet", "dit", "sd"
# export FUNCTION="inference"                        # "train", "inference", "explain"
export FUNCTION="train"
export DATA="chexpert"                            # "chexpert", "isic"

# For the baseline
export BACKBONE="efficientnet"                   # (str) Backbone for the classifier ('resnet' or 'efficientnet', 'vit', 'swin')
export VARIANT="efficientnet_b0"               # (str) Variant of the backbone ('resnet18', 'resnet50', 'efficientnet_b0', 'efficientnet_b4', 'swin_base_patch4_window7_224', 'vit_base_patch16_224', 'vit_small_patch16_224')


export SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Baseline classifier
if [[ "$MODEL" == "baseline" ]]; then
    if [[ "$FUNCTION" == "train" || "$FUNCTION" == "inference" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/baseline-classifier/${DATA}-classifier.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else 
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=baseline"
    fi
fi

# UNet model (supports training, inference, explain)
if [[ "$MODEL" == "unet" ]]; then
    if [[ "$FUNCTION" == "train" || "$FUNCTION" == "inference" || "$FUNCTION" == "explain" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/unet/${DATA}-unet.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=unet"
    fi
fi

# DiT model (supports training and inference)
if [[ "$MODEL" == "dit" ]]; then
    if [[ "$FUNCTION" == "train" || "$FUNCTION" == "inference" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/dit/${DATA}-dit.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=dit"
    fi
fi

# Stable Diffusion model
if [[ "$MODEL" == "sd" ]]; then
    if [[ "$FUNCTION" == "inference" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/stable-diffusion/sd-${DATA}-inference.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    elif [[ "$FUNCTION" == "train" ]]; then
        SCRIPT_PATH="$SCRIPTS_DIR/stable-diffusion/train.sh"
        if [[ -f "$SCRIPT_PATH" ]]; then
            echo "Note that SD model trains on both datasets at the same time"
            source "$SCRIPT_PATH"
        else
            echo "Error: $SCRIPT_PATH not found!"
        fi
    else
        echo "Error: FUNCTION=$FUNCTION is not supported for MODEL=sd"
    fi
fi