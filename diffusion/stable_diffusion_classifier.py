from comet_ml import Experiment, ExistingExperiment
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline, EulerDiscreteScheduler
from transformers import AutoModel, AutoTokenizer

import torch
import torch.nn as nn
from torch.special import expm1
import math
from accelerate import Accelerator
import os
import sys
from tqdm import tqdm
# from ema_pytorch import EMA
import time
import matplotlib.pyplot as plt

# helper
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

MODEL_IDS = {
            '1-1': "CompVis/stable-diffusion-v1-1",
            '1-2': "CompVis/stable-diffusion-v1-2",
            '1-3': "CompVis/stable-diffusion-v1-3",
            '1-4': "CompVis/stable-diffusion-v1-4",
            '1-5': "runwayml/stable-diffusion-v1-5",
            '2-0': "stabilityai/stable-diffusion-2-base",
            '2-1': "stabilityai/stable-diffusion-2-1-base"
        }

class StableDiffusionClassifier(nn.Module):
    def __init__(
        self,
        config: dict,
        label_to_text_mapper,
    ):
        super().__init__()
        # Training configuration
        self.config = config
        self.version = config.version
        self.model_path = config.model_path    
        if self.model_path is not None and self.model_path != "":
            if("radedit" not in self.model_path):
                assert os.path.exists(self.model_path), f"Model path {self.model_path} does not exist."
            self.model_id = self.model_path
        else:
            assert self.version in MODEL_IDS.keys()
            self.model_id = MODEL_IDS[self.version]

        self.mixed_precision = config.mixed_precision
        self.vae, self.tokenizer, self.text_encoder, self.unet, self.scheduler = self.get_sd_model(config.mixed_precision)

        self.label_to_text_mapper = label_to_text_mapper

        print("MODEL ID: ", self.model_id)

    def get_sd_model(self, dtype):
        assert dtype in ['fp32', 'fp16']

        if dtype == 'fp32':
            dtype = torch.float32
        elif dtype == 'fp16':
            dtype = torch.float16
        else:
            raise NotImplementedError
        
        if("radedit" in self.model_id):
            unet = UNet2DConditionModel.from_pretrained("microsoft/radedit", subfolder="unet")
            vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
            text_encoder = AutoModel.from_pretrained(
                "microsoft/BiomedVLP-BioViL-T",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/BiomedVLP-BioViL-T",
                model_max_length=128,
                trust_remote_code=True,
            )
            scheduler = EulerDiscreteScheduler(
                beta_schedule="scaled_linear", 
                prediction_type="epsilon", 
                timestep_spacing="trailing", 
                steps_offset=1
                )
            pipe = StableDiffusionPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                requires_safety_checker=False,
                feature_extractor=None,
            )
            pipe.enable_xformers_memory_efficient_attention()

            print("DTYPE: ", pipe.dtype)
            
        else:
            scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
            
            pipe = StableDiffusionPipeline.from_pretrained(self.model_id if self.model_path is None or self.model_path=="" else self.model_path, scheduler=scheduler, torch_dtype=dtype)
            pipe.enable_xformers_memory_efficient_attention()

            vae = pipe.vae
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            unet = pipe.unet

        # Loading directly from a checkpoint
        # from safetensors.torch import load_file
        # state_dict = load_file("/home/mila/p/parham.saremi/diffusion-classifier-new/experiments/stable-diffusion-medical/checkpoint-12500/unet/diffusion_pytorch_model.safetensors")
        # pipe.unet.load_state_dict(state_dict)

        pipe.to("cuda")

        return vae, tokenizer, text_encoder, unet, scheduler
    
    def get_scheduler_config(self, version):
        if version in {'1-1', '1-2', '1-3', '1-4', '1-5'}:
            config = {
                "_class_name": "EulerDiscreteScheduler",
                "_diffusers_version": "0.14.0",
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "interpolation_type": "linear",
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "set_alpha_to_one": False,
                "skip_prk_steps": True,
                "steps_offset": 1,
                "trained_betas": None
            }
        elif version in {'2-0', '2-1'}:
            config = {
                "_class_name": "EulerDiscreteScheduler",
                "_diffusers_version": "0.10.2",
                "beta_end": 0.012,
                "beta_schedule": "scaled_linear",
                "beta_start": 0.00085,
                "clip_sample": False,
                "num_train_timesteps": 1000,
                "prediction_type": "epsilon",
                "set_alpha_to_one": False,
                "skip_prk_steps": True,
                "steps_offset": 1,  # todo
                "trained_betas": None
            }
        else:
            raise NotImplementedError
        return config

    def encode_text_prompt(self, text):
        """
        Encode a text prompt using the selected encoder, if available.
        """
        text_input = self.tokenizer(text, padding="max_length",
                               max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        if("radedit" in self.model_id):
            text_embeddings = self.text_encoder(
                    text_input.input_ids.to(self.unet.device),
                    attention_mask=text_input.attention_mask.to(self.unet.device)
                )[0]
        else:
            text_embeddings = self.text_encoder(
                    text_input.input_ids.to(self.unet.device),
                )[0]
        
        return text_embeddings
    
    def diffuse(self, x, alpha_t, sigma_t):
        """
        Function to diffuse the input tensor x to a timepoint t with the given alpha_t and sigma_t.

        Args:
        x (torch.Tensor): The input tensor to diffuse.
        alpha_t (torch.Tensor): The alpha value at timepoint t.
        sigma_t (torch.Tensor): The sigma value at timepoint t.

        Returns:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        eps_t (torch.Tensor): The noise tensor at timepoint t.
        """
        eps_t = torch.randn_like(x)

        z_t = alpha_t * x + sigma_t * eps_t

        return z_t, eps_t
        
    def clip(self, x):
        """
        Function to clip the input tensor x to the range [-1, 1].

        Args:
        x (torch.Tensor): The input tensor to clip.

        Returns:
        x (torch.Tensor): The clipped tensor.
        """
        return torch.clamp(x, -1, 1)

    @torch.no_grad()
    def evaluate(
        self, 
        val_dataloader, 
        stop_idx=None,
        metrics=None,
    ):
        """
        A function to evaluate the model.

        Args:
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        stop_idx (int): The index to stop at.
        metrics (list): A list of metrics to evaluate.
        """
        
        val_samples = []
        batches = []

        all_outputs = []
        device = None

        # Make a progress bar
        progress_bar = tqdm(val_dataloader, desc="Evaluating")
        for idx, batch in enumerate(val_dataloader):
            progress_bar.update(1)

            batch = {k: v for k, v in batch.items()}

            x = batch["images"]
            p = batch["prompt"] if "prompt" in batch.keys() else None
            
            sample = self.classify(x, p, majority=self.config.majority_voting)
                        
            # Update the metrics
            if metrics is not None:
                for metric in metrics:
                    metric.update((sample, batch))

            val_samples.append(sample)
            batches.append(batch)

            if stop_idx is not None and idx == stop_idx:
                break
        # all_outputs = torch.cat(all_outputs, dim=0)

        # torch.save(all_outputs.cpu(), f"stats-isic-d{device.index}.pt")

        progress_bar.close()

        return val_samples, batches, metrics
    
    @torch.no_grad()
    def inference(
        self, 
        val_dataloader,
        metrics=None,
    ):
        """
        A function to perform inference.

        Args:
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        metrics (list): A list of metrics to evaluate.

        Returns:
        metric_output (list): The output of the metrics. (If metrics is not None)
        val_samples (list): The validation samples.
        batches (list): The batches
        """

        # Make directory for saving images
        inference_image_path = os.path.join(self.config.experiment_path, "inference_images/")
        os.makedirs(inference_image_path, exist_ok=True)

        # Make accelerator wrapper
        accelerator = Accelerator()

        self.unet, self.text_encoder, self.tokenizer, self.vae, val_dataloader = accelerator.prepare( 
            self.unet, self.text_encoder, self.tokenizer, self.vae, val_dataloader
        )

        if metrics is not None:
            for metric in metrics:
                metric.set_device(accelerator.device)

        val_samples, batches, metrics = self.evaluate(
                                            val_dataloader, 
                                            metrics=metrics,
                                            stop_idx=self.config.evaluation_batches,
                                        )
        
        metric_output = []
        if metrics is not None:
            for metric in metrics:
                metric.sync_across_processes(accelerator)
                metric_output.append(metric.get_output())

        return (metric_output, val_samples, batches) if metrics is not None else (val_samples, batches)
    
    @torch.no_grad()
    def classify(self, x, text=None, majority=True):
        """
        Function to classify the input tensor x.

        Args:
        x (torch.Tensor): The input tensor to classify.
        text (str): The text prompt to use.
        majority (bool): Whether to use majority voting or average voting.

        Returns:
        classes (torch.Tensor): The predicted classes.
        """
        assert len(self.config.evaluation_per_stage) == self.config.n_stages, "Number of evaluations per stage must match the number of stages."
        assert len(self.config.n_keep_per_stage) == self.config.n_stages, "Number of classes to keep per stage must match the number of stages."
        assert self.config.n_keep_per_stage[-1] == 1, "Only one class should be selected at the end of the classification process."

        evaluation_per_stage = [0] + self.config.evaluation_per_stage

        BS = x.shape[0]

        if self.mixed_precision == 'fp16':
            x = x.half()
        x0 = self.vae.encode(x).latent_dist.mean

        x0 *= 0.18215

        errors = torch.full((BS, self.config.classes, evaluation_per_stage[-1]), torch.inf).to(x.device) # Store the errors for each class Originally set to 10000. 

        classes = torch.arange(self.config.classes).repeat(BS, 1).to(x.device) # of shape (BS, classes)
        
        for i in range(self.config.n_stages):
            # Get the start and end indices for the current stage
            stage_start = evaluation_per_stage[i]
            stage_end = evaluation_per_stage[i + 1]

            for j in tqdm(range(stage_start, stage_end)):
                # Sampling timepoint t and image at time t
                t = torch.randint(1,999, (BS,))
                t_input = t.to(x.device)
                if self.mixed_precision == 'fp16':
                    t_input = t_input.half()

                alpha_t = (self.scheduler.alphas_cumprod[t] ** 0.5).view(-1, 1, 1, 1).to(x.device)
                sigma_t = ((1 - self.scheduler.alphas_cumprod[t]) ** 0.5).view(-1, 1, 1, 1).to(x.device)
                z_t, eps_t = self.diffuse(x0, alpha_t, sigma_t)

                # Get the errors for each class
                for c in range(classes.shape[1]):
                    labels = classes[:, c]
                    text = self.label_to_text_mapper(labels)
                    
                    text_embeddings = self.encode_text_prompt(text)
                    
                    pred = self.unet(z_t, t_input, encoder_hidden_states=text_embeddings).sample
                    eps_pred = pred

                    error_c = torch.norm((eps_pred - eps_t).view(BS, -1), dim=1, p=2)**2

                    batch_indices = torch.arange(BS, device=x.device)
                    errors[batch_indices, classes[:, c], j] = error_c
                    
            if not majority:
                # Average voting-based selection
                num_keep = self.config.n_keep_per_stage[i]
                end_of_stage_errors = errors[:, :, :stage_end].mean(dim=2) # Average the errors until now
                _, keep_indices = torch.topk(end_of_stage_errors, num_keep, dim=1, largest=False) # Keep the classes with the lowest errors
                classes = keep_indices # of shape (BS, num_keep): The indices of the classes to keep
            else:
                num_keep = self.config.n_keep_per_stage[i]
                # Majority voting-based selection
                end_of_stage_votes = errors[:, :, :stage_end].argmin(dim=1) # Get the class with the lowest error
                votes = torch.zeros(BS, self.config.classes).to(x.device)
                votes.scatter_add_(
                    1,  # Scatter along the class dimension
                    end_of_stage_votes,  # Indices of classes (Shape: (BS, stage_end))
                    torch.ones_like(end_of_stage_votes, dtype=votes.dtype, device=x.device)  # Add 1 to the corresponding class
                )

                _, keep_indices = torch.topk(votes, num_keep, dim=1, largest=True) 
                classes = keep_indices # of shape (BS, num_keep): The indices of the classes to keep

        return classes[:, 0]
    
