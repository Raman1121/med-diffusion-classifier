from comet_ml import Experiment, ExistingExperiment
import torch
import torch.nn as nn
from torch.special import expm1
import math
from accelerate import Accelerator
import os
import sys
from tqdm import tqdm
from ema_pytorch import EMA
import time

# helper
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

class DiffusionClassifier(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module,
        config: dict,
    ):
        super().__init__()

        # Training configuration
        self.config = config

        # Training objective
        pred_param = self.config.pred_param
        assert pred_param in ['v', 'eps'], "Invalid prediction parameterization. Must be 'v' or 'eps'"
        self.pred_param = pred_param

        # Sampling schedule
        schedule = self.config.schedule
        assert schedule in ['cosine', 'shifted_cosine'], "Invalid schedule. Must be 'cosine' or 'shifted_cosine'"
        if schedule == 'cosine':
            self.schedule = self.logsnr_schedule_cosine
        elif schedule == 'shifted_cosine':
            self.schedule = self.logsnr_schedule_cosine_shifted
        self.noise_d = self.config.noise_d
        self.image_d = self.config.image_size

        # Classifier-free guidance scale
        self.cfg_w = self.config.cfg_w

        # Model
        assert isinstance(backbone, nn.Module), "Model must be an instance of torch.nn.Module."
        self.model = backbone

        # EMA version of the model
        self.ema = EMA(
            self.model,
            beta=config.ema_beta,
            update_after_step=config.ema_warmup,
            update_every=config.ema_update_freq,
        )

        # Optional encoder setup
        self.encoder_type = self.config.encoder_type
        if self.encoder_type == 't5':
            from transformers import T5EncoderModel, T5Tokenizer
            self.encoder = T5EncoderModel.from_pretrained("t5-base")
            self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
            self.null_token = self.tokenizer.pad_token_id
        elif self.encoder_type == 'nn':
            classes = self.config.classes+1
            embedding_dim = backbone.config.encoder_hid_dim
            self.encoder = nn.Embedding(classes, embedding_dim)
            self.tokenizer = None  # Not required for embeddings
            self.null_token = self.config.classes  # Placeholder for null token
        elif self.encoder_type == 'DiT':
            self.tokenizer = None
            self.encoder = None
            self.null_token = self.config.classes  # Placeholder for null token

        # Lock the parameters of the encoder - Disable for end-to-end training
        if self.encoder_type in ['t5']:
            self.encoder.requires_grad_(False)

        # Print the number of parameters in self.model
        print(f"Parameter count: {sum(p.numel() for p in self.model.parameters())}")

        self.flash_attention = self.config.flash_attention

    def encode_text_prompt(self, text):
        """
        Encode a text prompt using the selected encoder, if available.
        """
        if self.encoder_type == 'nn':
            embeddings = self.encoder(text)
            embeddings.unsqueeze_(1)
        elif self.encoder_type == 'DiT':
            # Leave class labels as they are
            embeddings = text
        else:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs.to(self.encoder.device)
            with torch.no_grad():
                embeddings = self.encoder(**inputs).last_hidden_state
        return embeddings
    
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

    def logsnr_schedule_cosine(self, t, logsnr_min=-15, logsnr_max=15):
        """
        Function to compute the logSNR schedule at timepoint t with cosine:

        logSNR(t) = -2 * log (tan (pi * t / 2))

        Taking into account boundary effects, the logSNR value at timepoint t is computed as:

        logsnr_t = -2 * log(tan(t_min + t * (t_max - t_min)))

        Args:
        t (int): The timepoint t.
        logsnr_min (int): The minimum logSNR value.
        logsnr_max (int): The maximum logSNR value.

        Returns:
        logsnr_t (float): The logSNR value at timepoint t.
        """
        logsnr_max = logsnr_max + math.log(self.noise_d / self.image_d)
        logsnr_min = logsnr_min + math.log(self.noise_d / self.image_d)
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))

        logsnr_t = -2 * log(torch.tan((t_min + t * (t_max - t_min)).clone().detach()))

        return logsnr_t

    def logsnr_schedule_cosine_shifted(self, t):
        """
        Function to compute the logSNR schedule at timepoint t with shifted cosine:

        logSNR_shifted(t) = logSNR(t) + 2 * log(noise_d / image_d)

        Args:
        t (int): The timepoint t.

        Returns:
        logsnr_t_shifted (float): The logSNR value at timepoint t.
        """
        logsnr_t = self.logsnr_schedule_cosine(t)
        logsnr_t_shifted = logsnr_t + 2 * math.log(self.noise_d / self.image_d)

        return logsnr_t_shifted
        
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
    def ddpm_sampler_step(self, z_t, pred, u_pred, logsnr_t, logsnr_s):
        """
        Function to perform a single step of the DDPM sampler.

        Args:
        z_t (torch.Tensor): The diffused tensor at timepoint t.
        pred (torch.Tensor): The predicted value from the model (v or eps).
        u_pred (torch.Tensor): The unconditional prediction from the model.
        logsnr_t (float): The logSNR value at timepoint t.
        logsnr_s (float): The logSNR value at the sampling timepoint s.

        Returns:
        z_s (torch.Tensor): The diffused tensor at sampling timepoint s.
        """
        c = -expm1(logsnr_t - logsnr_s)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t))
        alpha_s = torch.sqrt(torch.sigmoid(logsnr_s))
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t))
        sigma_s = torch.sqrt(torch.sigmoid(-logsnr_s))

        w = self.cfg_w
        pred = (1+w)*pred - w*u_pred # cfg_pred = (1+w)pred - w*u_pred
        if self.pred_param == 'v':
            x_pred = alpha_t * z_t - sigma_t * pred
        elif self.pred_param == 'eps':
            x_pred = (z_t - sigma_t * pred) / alpha_t

        x_pred = self.clip(x_pred)

        mu = alpha_s * (z_t * (1 - c) / alpha_t + c * x_pred)
        variance = (sigma_s ** 2) * c

        return mu, variance
    
    @torch.no_grad()
    def sample(self, x, text=None, from_t=1):
        """
        Standard DDPM sampling procedure. Begun by sampling z_T ~ N(0, 1)
        and then repeatedly sampling z_s ~ p(z_s | z_t)

        Args:
        x (torch.Tensor): An input tensor that is of the desired shape.
        text (torch.Tensor): The text prompt tensor.
        from_t (int): The timepoint to start sampling from (default is 1).

        Returns:
        x_pred (torch.Tensor): The predicted tensor.
        """
        if from_t == 1:
            z_t = torch.randn(x.shape).to(x.device)
        else:
            t = torch.ones(x.shape[0]) * from_t
            logsnr_t = self.schedule(t).to(x.device)
            alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
            sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
            z_t, _ = self.diffuse(x, alpha_t, sigma_t)

        # Get embeddings (and null embeddings) if text is provided
        if text is not None and self.encoder_type is not None:
            text_embeddings = self.encode_text_prompt(text)
            text_embeddings = text_embeddings.to(x.device)

            null_tokens = torch.full_like(text, self.null_token)
            null_embeddings = self.encode_text_prompt(null_tokens)
            null_embeddings = null_embeddings.to(x.device)
        else:
            text_embeddings = None
            null_embeddings = None

        # Create evenly spaced steps, e.g., for sampling_steps=5 -> [1.0, 0.8, 0.6, 0.4, 0.2]
        steps = torch.linspace(from_t, 0.0, self.config.sampling_steps + 1)

        for i in range(len(steps) - 1):  # Loop through steps, but stop before the last element
            
            u_t = steps[i]  # Current step
            u_s = steps[i + 1]  # Next step

            logsnr_t = self.schedule(u_t).to(x.device).unsqueeze(0)
            logsnr_s = self.schedule(u_s).to(x.device).unsqueeze(0)

            # Conditional sample
            pred = self.ema(
                z_t, 
                logsnr_t,
                encoder_hidden_states=text_embeddings
                )

            # Unconditional sample
            u_pred = self.ema(
                z_t, 
                logsnr_t,
                encoder_hidden_states=null_embeddings
                )

            mu, variance = self.ddpm_sampler_step(z_t, pred, u_pred, logsnr_t.clone().detach(), logsnr_s.clone().detach())
            z_t = mu + torch.randn_like(mu) * torch.sqrt(variance)

        # Final step
        logsnr_1 = self.schedule(steps[-2]).to(x.device).unsqueeze(0)
        logsnr_0 = self.schedule(steps[-1]).to(x.device).unsqueeze(0)

        # Conditional sample
        pred = self.ema(
            z_t, 
            logsnr_1,
            encoder_hidden_states=text_embeddings
            )

        # Unconditional sample
        u_pred = self.ema(
            z_t,
            logsnr_1,
            encoder_hidden_states=null_embeddings
            )

        x_pred, _ = self.ddpm_sampler_step(z_t, pred, u_pred, logsnr_1.clone().detach(), logsnr_0.clone().detach())
        
        x_pred = self.clip(x_pred)

        return x_pred
    
    def loss(self, x, text=None):
        """
        A function to compute the loss of the model. The loss is computed as the mean squared error
        between the predicted noise tensor and the true noise tensor. Various prediction parameterizations
        imply various weighting schemes as outlined in Kingma et al. (2023)

        Args:
        x (torch.Tensor): The input tensor.
        text (torch.Tensor): The text prompt tensor.

        Returns:
        loss (torch.Tensor): The loss value.
        """
        t = torch.rand(x.shape[0])

        if text is not None and self.encoder_type is not None:
            text_embeddings = self.encode_text_prompt(text)
            text_embeddings = text_embeddings.to(x.device)
        else:
            text_embeddings = None

        logsnr_t = self.schedule(t).to(x.device)
        alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
        z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)
        pred = self.model(
            x=z_t, 
            noise_labels=logsnr_t,
            encoder_hidden_states=text_embeddings,
            )

        if self.pred_param == 'v':
            eps_pred = sigma_t * z_t + alpha_t * pred
        else: 
            eps_pred = pred

        # Apply min-SNR weighting (https://arxiv.org/pdf/2303.09556)
        snr = torch.exp(logsnr_t).clamp_(max = 5)
        if self.pred_param == 'v':
            weight = 1 / (1 + snr)
        else:
            weight = 1 / snr
        minsnr_weight = weight.view(-1, 1, 1, 1)

        # Get the absolute error
        error = eps_pred - eps_t

        loss = torch.mean(minsnr_weight * (error) ** 2)

        return loss
    
    def train_loop(
        self, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        lr_scheduler, 
        metrics=None,
        checkpoint_metric=None,
        plot_function=None,
    ):
        """
        A function to train the model.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        metrics (list): A list of metrics to evaluate.
        checkpoint_metric (str): The metric to use for saving the best checkpoint.
        plot_function (function): The function to use for plotting the samples.

        Returns:
        None
        """

        # Initialize CometML experiment
        experiment = None

        # Initialize accelerator
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=self.config.experiment_path,
        )
        
        # Prepare the model, optimizer, dataloaders, and learning rate scheduler
        model, ema, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare( 
            self.model, self.ema, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        if self.encoder_type == 'nn': # Learnable embeddings require separate preparation than pretrained models
            self.encoder = accelerator.prepare(self.encoder)

        # Ensure metrics are on the correct device
        if metrics is not None:
            checkpoint_tracker = {'value': 0.0, 'save_flag': False}
            for metric in metrics:
                metric.set_device(accelerator.device)
        else:
            checkpoint_tracker = None

        # Check if resume training is enabled
        if self.config.resume:
            checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
            start_epoch, checkpoint_tracker, experiment_key = self.load_checkpoint(checkpoint_path, accelerator)
            if experiment_key is not None and self.config.use_comet and accelerator.is_main_process:
                experiment = ExistingExperiment(
                    previous_experiment=experiment_key,
                    api_key=self.config.comet_api_key,
                )
        else: # Set up fresh experiment
            if self.config.use_comet and accelerator.is_main_process:
                experiment = Experiment(
                    api_key=self.config.comet_api_key,
                    project_name=self.config.comet_project_name,
                    workspace=self.config.comet_workspace,
                )
                experiment.set_name(self.config.comet_experiment_name)
                experiment.log_asset(os.path.join(self.config.experiment_path, 'train.py'), 'train.py')
                experiment.log_asset(os.path.join(self.config.project_root, 'train.sh'), 'train.sh')
                experiment.log_other("GPU Model", torch.cuda.get_device_name(0))
                experiment.log_other("Python Version", sys.version)
            start_epoch = 0

        # Train!
        if accelerator.is_main_process:
            print(self.config.__dict__)

        for epoch in range(start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()

            model.train()

            for _, batch in enumerate(train_dataloader):

                with accelerator.accumulate(model):
                    x = batch["images"]
                    p = batch["prompt"] if "prompt" in batch.keys() else None

                    # Stochastically drop out the conditions with probability p_drop
                    p_drop = 0.15
                    if p is not None:
                        # Sometimes replace the prompt with a null token
                        mask = torch.rand_like(p.float()) < p_drop
                        p = torch.where(mask, torch.full_like(p, self.null_token), p).long()

                    loss = self.loss(x, p)
                    loss = loss.to(next(model.parameters()).dtype)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        model_params = dict(model.named_parameters())
                        all_params = {**model_params}
                        accelerator.clip_grad_norm_(all_params.values(), max_norm=1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    self.ema.update()

            epoch_elapsed = time.time() - epoch_start_time
            if accelerator.is_main_process:
                print(f"Epoch {epoch}/{self.config.num_epochs}: {epoch_elapsed:.2f} s.")

                # Log the loss to CometML
                if experiment is not None:
                    experiment.log_metric("loss", loss.item(), epoch=epoch)

            # Run an evaluation on validation set
            if epoch % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                val_evaluation_start_time = time.time()

                model.eval()

                # Make directory for saving images
                training_images_path = os.path.join(self.config.experiment_path, "training_images/")

                # TODO: cleanup and consider multiple metrics
                val_samples, batches, _ = self.evaluate(
                                            val_dataloader, 
                                            stop_idx=self.config.evaluation_batches,
                                            metrics=None,
                                        )
                
                _, _, metrics = self.evaluate(
                                    val_dataloader, 
                                    stop_idx=self.config.evaluation_batches,
                                    metrics=metrics,
                                    classification=True
                                )
                
                # Use the provided plot_function to plot the samples
                if plot_function is not None:
                    image_path = plot_function(
                        output_dir=training_images_path,
                        batches=batches,
                        samples=val_samples, 
                        epoch=epoch, 
                        process_idx=accelerator.state.process_index
                    )

                if metrics is not None:
                    for metric in metrics:
                        metric.sync_across_processes(accelerator)
                        metric_output = metric.get_output()

                        if (checkpoint_metric is not None) and (metric.name == checkpoint_metric):
                            if metric_output[metric.name].item() > checkpoint_tracker['value']:
                                checkpoint_tracker['value'] = metric_output[metric.name].item()
                                checkpoint_tracker['save_flag'] = True

                        if accelerator.is_main_process:
                            if experiment is not None: 
                                log_output = {f"val_{metric_name}": value for metric_name, value in metric_output.items()}
                                experiment.log_metrics(log_output, step=epoch)
                                experiment.log_image(name=f"Sample at epoch {epoch}", image_data=image_path)

                            print(metric_output)

                        metric.reset()
                
                # Print some statistics, save (best) checkpoint
                val_evaluation_elapsed = time.time() - val_evaluation_start_time
                if accelerator.is_main_process:
                    if (checkpoint_metric is not None): 
                        self.save_checkpoint(accelerator, epoch, experiment, checkpoint_tracker)
                    else:
                        self.save_checkpoint(accelerator, epoch, experiment)

                    print(f"Val evaluation time: {val_evaluation_elapsed:.2f} s.")

                # Reset flags
                checkpoint_tracker['save_flag'] = False
                model.train()

    @torch.no_grad()
    def evaluate(
        self, 
        val_dataloader, 
        stop_idx=None,
        metrics=None,
        classification=False,
        from_t=1
    ):
        """
        A function to evaluate the model.

        Args:
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        stop_idx (int): The index to stop at.
        metrics (list): A list of metrics to evaluate.
        classification (bool): Whether to perform classification.
        from_t (int): The timepoint to start sampling from (default is 1).
        """
        
        val_samples = []
        batches = []

        # Make a progress bar
        progress_bar = tqdm(val_dataloader, desc="Evaluating")
        for idx, batch in enumerate(val_dataloader):
            progress_bar.update(1)

            batch = {k: v for k, v in batch.items()}

            x = batch["images"]
            p = batch["prompt"] if "prompt" in batch.keys() else None
            
            sample = self.classify(x, p, majority=self.config.majority_voting) if classification else self.sample(x, p, from_t)
                        
            # Update the metrics
            if metrics is not None:
                for metric in metrics:
                    metric.update((sample, batch))

            val_samples.append(sample)
            batches.append(batch)

            if stop_idx is not None and idx == stop_idx:
                break

        progress_bar.close()

        return val_samples, batches, metrics
    
    # TODO - fix parameters because some of them are just config.something
    @torch.no_grad()
    def inference(
        self, 
        optimizer, 
        train_dataloader, 
        val_dataloader,
        lr_scheduler, 
        metrics=None,
        plot_function=None,
        classification=False,
        from_t=1,
        checkpoint_folder="checkpoints"
    ):
        """
        A function to perform inference.

        Args:
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        train_dataloader (torch.utils.data.DataLoader): The training dataloader.
        val_dataloader (torch.utils.data.DataLoader): The validation dataloader.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        metrics (list): A list of metrics to evaluate.
        plot_function (function): The function to use for plotting the samples.
        classification (bool): Whether to perform classification.
        from_t (int): The timepoint to start sampling from (default is 1).
        checkpoint_folder (str): The folder to load the checkpoint from. Default is "checkpoints" which will use the checkpoints folder in the experiment folder. Otherwise should provide absolute path.
        """

        # Make directory for saving images
        inference_image_path = os.path.join(self.config.experiment_path, "inference_images/")
        os.makedirs(inference_image_path, exist_ok=True)

        # Make accelerator wrapper
        accelerator = Accelerator()

        model, ema, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare( 
            self.model, self.ema, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        if self.encoder_type == 'nn': # Learnable embeddings require separate preparation than pretrained models
            self.encoder = accelerator.prepare(self.encoder)
        
        # Load most recent checkpoint
        if checkpoint_folder == "checkpoints" or checkpoint_folder == "" or checkpoint_folder is None:
            checkpoint_path = os.path.join(self.config.experiment_path, "checkpoints")
        else:
            checkpoint_path = checkpoint_folder

        self.load_checkpoint(checkpoint_path, accelerator)

        if metrics is not None:
            for metric in metrics:
                metric.set_device(accelerator.device)

        # TODO - metrics, classification, image sampling needs to be in a state machine
        if self.flash_attention:
            self.ema.half()
        model.eval()
        val_samples, batches, metrics = self.evaluate(
                                                    val_dataloader, 
                                                    metrics=metrics,
                                                    stop_idx=self.config.evaluation_batches,
                                                    classification=classification,
                                                    from_t=from_t
                                                )
        
        metric_output = []
        if metrics is not None:
            for metric in metrics:
                metric.sync_across_processes(accelerator)
                metric_output.append(metric.get_output())

        # Use the provided plot_function to plot the samples
        if plot_function is not None and not classification:
            image_path = plot_function(
                output_dir=inference_image_path,
                batches=batches,
                samples=val_samples, 
                epoch=0, 
                process_idx=accelerator.state.process_index
            )

        return (metric_output, val_samples, batches) if metrics is not None else (val_samples, batches)
    
    @torch.no_grad()
    def classify(self, x, text=None, majority=True):
        """
        A function to perform classification.

        Args:
        x (torch.Tensor): The input tensor.
        text (torch.Tensor): The text prompt tensor.
        majority (bool): Whether to use majority voting.

        Returns:
        classes (torch.Tensor): The predicted classes.
        """
        assert self.encoder_type is not None, "Encoder must be provided for classification."
        assert len(self.config.evaluation_per_stage) == self.config.n_stages, "Number of evaluations per stage must match the number of stages."
        assert len(self.config.n_keep_per_stage) == self.config.n_stages, "Number of classes to keep per stage must match the number of stages."
        assert self.config.n_keep_per_stage[-1] == 1, "Only one class should be selected at the end of the classification process."

        evaluation_per_stage = [0] + self.config.evaluation_per_stage

        BS = x.shape[0]

        errors = torch.full((BS, self.config.classes, evaluation_per_stage[-1]), torch.inf).to(x.device) # Store the errors for each class Originally set to 10000. 

        classes = torch.arange(self.config.classes).repeat(BS, 1).to(x.device) # of shape (BS, classes)
        
        for i in range(self.config.n_stages):
            # Get the start and end indices for the current stage
            stage_start = evaluation_per_stage[i]
            stage_end = evaluation_per_stage[i + 1]

            for j in tqdm(range(stage_start, stage_end)):
                # Sampling timepoint t and image at time t
                t = torch.rand(BS)
                logsnr_t = self.schedule(t).to(x.device)
                alpha_t = torch.sqrt(torch.sigmoid(logsnr_t)).view(-1, 1, 1, 1).to(x.device)
                sigma_t = torch.sqrt(torch.sigmoid(-logsnr_t)).view(-1, 1, 1, 1).to(x.device)
                z_t, eps_t = self.diffuse(x, alpha_t, sigma_t)

                # Get the errors for each class
                for c in range(classes.shape[1]):
                    text = classes[:, c]
                    text_embeddings = self.encode_text_prompt(text)
                    text_embeddings = text_embeddings.to(x.device)

                    if self.flash_attention:
                        z_t = z_t.half()
                        logsnr_t = logsnr_t.half()
                        if self.encoder_type == 'nn':
                            text_embeddings = text_embeddings.half()

                    pred = self.ema(
                        x=z_t, 
                        noise_labels=logsnr_t,
                        encoder_hidden_states=text_embeddings,
                        )
                    
                    if self.pred_param == 'v':
                        eps_pred = sigma_t * z_t + alpha_t * pred
                    else: 
                        eps_pred = pred

                    error_c = torch.norm((eps_pred - eps_t).view(BS, -1), dim=1, p=2)**2

                    batch_indices = torch.arange(BS, device=x.device)
                    errors[batch_indices, classes[:, c], j] = error_c
            
            num_keep = self.config.n_keep_per_stage[i]

            if not majority:
                end_of_stage_errors = errors[:, :, :stage_end].mean(dim=2) # Average the errors until now
                _, keep_indices = torch.topk(end_of_stage_errors, num_keep, dim=1, largest=False) # Keep the classes with the lowest errors
                classes = keep_indices # of shape (BS, num_keep): The indices of the classes to keep
            else:
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
    
    def save_checkpoint(self, accelerator: Accelerator, epoch, experiment, checkpoint_tracker=None):
        """
        Saves the model checkpoint.

        Args:
        accelerator (accelerate.Accelerator): The Accelerator object.
        epoch (int): The current epoch.
        experiment (comet_ml.Experiment): The CometML experiment object.
        checkpoint_tracker (dict): Contains whether it's the best model or not with the metric value.
        """
        checkpoint_dir = os.path.join(self.config.experiment_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists

        # Save accelerator state
        accelerator.save_state(output_dir=checkpoint_dir)

        # Save experiment state
        experiment_state = {
            'epoch': epoch + 1,
            'best_metric': checkpoint_tracker['value'] if checkpoint_tracker is not None else None,
            'experiment_key': experiment.get_key() if experiment is not None else None,
        }

        # Save new checkpoint
        latest_exp_state_path = os.path.join(checkpoint_dir, "experiment_state.pth")
        torch.save(experiment_state, latest_exp_state_path)

        print(f"Checkpoint saved to {latest_exp_state_path}")

        # Save best checkpoint if necessary
        best = checkpoint_tracker['save_flag'] if checkpoint_tracker is not None else False
        if best:
            best_checkpoint_dir = os.path.join(self.config.experiment_path, "best_checkpoint")
            os.makedirs(best_checkpoint_dir, exist_ok=True)  # Ensure directory exists

            # Save accelerator state
            accelerator.save_state(output_dir=best_checkpoint_dir)

            best_exp_state_path = os.path.join(best_checkpoint_dir, "experiment_state.pth")
            torch.save(experiment_state, best_exp_state_path)
            print(f"Best checkpoint saved to {best_exp_state_path}")
    
    def load_checkpoint(self, checkpoint_path, accelerator: Accelerator):
        """
        Loads the model checkpoint.

        Args:
        checkpoint_path (str): The path to the checkpoint file.
        accelerator (accelerate.Accelerator): The Accelerator object.

        Returns:
        tuple: The epoch, best metric (if available), and experiment.
        """
        # Restore the accelerator state
        accelerator.load_state(input_dir=checkpoint_path)

        # Get the experiment state path
        experiment_state_path = os.path.join(checkpoint_path, "experiment_state.pth")

        # Load the checkpoint file on CPU
        checkpoint = torch.load(experiment_state_path, map_location='cpu', weights_only=False)

        # Restore the epoch to resume training from
        epoch = checkpoint['epoch']

        # Restore the best metric value
        try:
            best_metric = {
                "value": checkpoint['best_metric'],
                "save_flag": False
            }
        except:
            best_metric = None

        # Optionally, resume the experiment from Comet (if using Comet for tracking)
        experiment_key = checkpoint['experiment_key']

        print(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {epoch}. Best metric {best_metric}")

        return epoch, best_metric, experiment_key