from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
import argparse
import os, sys
import os.path as path
from datetime import datetime
import logging
from data.dataloader_nc_v2 import dataloader_superres
#from data.dataloader_nc import dataloader_superres
from models.models import  ERAencoder, ERAdecoder, ERA5Upscaler, ERA5UpscalerV2
from models.mae import MAE, MAE_decoder
from utils.forecast_metrics import reconstruct_image_reduced, plot_reconstruction, reconstruct_image
from utils.arguments import TrainingParam, ModelParam, OptParam
from torch.utils.data import DataLoader
import itertools
from itertools import cycle
import einops
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields
import pickle
from captum.attr import IntegratedGradients

logging.basicConfig(level=logging.DEBUG)

train_dataset_ = dataloader_superres( \
                "/pscratch/sd/j/jderm/ERA5_golden/",
                "/pscratch/sd/j/jderm/HRRR_npy/",
                "/global/common/software/m4506/s2s_mae/training/s2s_tar/goldstandard/gold_npy_buoy_test_sorted.csv",
                pretraining = False,
                use_npy=True,
                two_hours=False)
patch_mask_indices = torch.tensor(train_dataset_.patch_mask_indices)
global_baseline = torch.rand(torch.load("/pscratch/sd/j/jderm/median_input.pt").shape)


# Count the total number of parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())



def integrated_gradients_for_spatial_output(model, input_tensor, target_h, target_w, target_dim=None):
    """
    Apply Integrated Gradients to explain predictions at specific spatial location
    
    Args:
        model: Your PyTorch model
        input_tensor: Input to the model (e.g., tokens or embeddings)
        target_h: Height index to explain
        target_w: Width index to explain
        target_dim: Specific dimension to explain (if None, will use sum/mean across dimensions)
    
    Returns:
        Attributions with respect to input
    """
    # Define forward function that extracts the desired output location
    def forward_func(x1):
        b = x1.shape[0]
        x2 = torch.zeros(b, 910, 200,requires_grad=False).to(x1.device)
        _,p,d = x2.shape
        outputs,_,_ = model(x1, x2, two_hours=False, decoder_masking_ratio=1)  # Shape: [batch_size, h, w, dim]
        coord_int = einops.rearrange(torch.arange(0,1742),'(h w) -> h w',h=67,w=26)[target_h,target_w]
        coord_int = torch.argmax((patch_mask_indices==coord_int).int())

        if target_dim is not None:
            # Extract specific dimension at target location
            return outputs[:, coord_int, target_dim]
        else:
            # Use all dimensions at target location (summed or other aggregation)
            return outputs[:, coord_int,:].sum(dim=-1)
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(forward_func)
    
    # Define baseline (typically zeros or some reference value)
    baseline0 = global_baseline.to(input_tensor.device)

    #        reconstruction, sample_y2,_ = self.model(sample_x, sample_y, =True, decoder_masking_ratio=self.decoder_mask_prob) # img_tminus1 = sample_x_minus1)
    
    # Calculate attributions
    attributions = ig.attribute(
        input_tensor, 
        baseline0, 
        n_steps=250,
        internal_batch_size=10,
        return_convergence_delta=True
    )
    
    # attributions[0] contains the actual attributions
    # attributions[1] contains convergence deltas which can be used to verify reliability
    return attributions


hrrr_mean_var = np.array([ \
[1.5682458877563477,  3.0335888862609863],
[-3.9326555728912354,  4.704221725463867]])
#buoy
hrrr_mean_var = np.array([ \
 [ 1.62887898,  3.81834122],
 [-4.01411624,  6.61316559]])

class trainer():
    def __init__(self, device, checkpoint_path, experiment_string, pretrain, finetune1, finetune2, npy ,epochs, batch_size, dataparallel,
                 era5_path, hrrr_path, train_file, val_file, grad_accumulation_steps, validate,
                 encoder_learning_rate, encoder_weight_decay, decoder_learning_rate, decoder_weight_decay,
                 encoder_depth, encoder_dim, encoder_channels, encoder_heads, encoder_mlp_dim, encoder_num_registers, 
                 decoder_depth, decoder_dim, decoder_channels, decoder_heads, decoder_mlp_dim,
                 encoder_masking_ratio, decoder_masking_ratio, grad_norm, seed, beta1, beta2, eps):

        self.grad_accumulation_steps = grad_accumulation_steps

        self.accelerator = Accelerator(gradient_accumulation_steps=self.grad_accumulation_steps, log_with="tensorboard",project_dir=f"/pscratch/sd/j/jderm/tb_logdir_july/{experiment_string}")
        self.accelerator.init_trackers(experiment_string)


        logging.info(f"num_process:{self.accelerator.num_processes}")

        #set_seed(self.accelerator.process_index + seed)
        set_seed(seed)

        tracker = self.accelerator.get_tracker("tensorboard")
        tracker.store_init_configuration({"lr":str(encoder_learning_rate)})

        if not path.exists(train_file):
            logging.error("train file does not exist")
            sys.exit(0)

        if not path.exists(val_file):
            logging.error("val file does not exist")
            sys.exit(0)

        if not path.exists(era5_path):
            logging.error("ERA5 path specified does not exist")
            sys.exit(0)

        if not path.exists(hrrr_path):
            logging.error("HRRR path specified does not exist")
            sys.exit(0)

        self.u80_list = list()
        self.v80_list = list()
        self.uv_iter = 0

        self.load_checkpoint_bool = False
        self.legacy_checkpoint_bool = False

        if not not checkpoint_path:
            self.load_checkpoint_bool = True

            if ".pt" in checkpoint_path:
                self.legacy_checkpoint_bool = True

        self.pretrain  = pretrain
        self.finetune1 = finetune1
        self.finetune2 = finetune2
        self.npy = npy
        self.warmup = True
        self.warmup_steps = 2000 
        self.eval_masking_ratio = 0.60
        self.eval_masking_strategy = 0
        self.first_finetune_epoch = False
        self.dataparallel = dataparallel
        self.encoder_masking_ratio = encoder_masking_ratio
        self.decoder_masking_ratio = decoder_masking_ratio
        self.grad_norm = grad_norm
        self.batch_size = batch_size
        self.n_epochs = epochs
        self.experiment_string = experiment_string

        self.eps = eps
        self.beta1 = beta1
        self.beta2 = beta2

        self.save_interval = 5
        self.train_log_interval = 20
        self.val_log_interval = self.train_log_interval*4
        self.gpu_interval = self.train_log_interval*16

        self.i_epoch = 0 #100 #65
        self.i_batch = 0
        self.i_step = 0 #72550 #24120 #100000 #62320
        self.global_loss = None
        self.total_norm = None


        self.checkpoint_epoch = 0
        self.validate = validate

        if isinstance(device, str):
           if device =='cpu':
               device = torch.device("cpu")
           elif device =='accelerate':
               device = self.accelerator.device
           else:
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        encoder_learning_rate *= self.accelerator.num_processes
        decoder_learning_rate *= self.accelerator.num_processes

        encoder_learning_rate *= self.grad_accumulation_steps
        decoder_learning_rate *= self.grad_accumulation_steps

        self.encoder = ERAencoder(image_size=(240,440),
	       patch_size=(10,10), 
	       num_classes=192,
           channels=encoder_channels,
	       dim=encoder_dim,
	       depth=encoder_depth,
	       heads=encoder_heads,
           num_registers=encoder_num_registers,
	       mlp_dim=encoder_mlp_dim)

        if self.pretrain:
            self.model = MAE(encoder=self.encoder, decoder_dim=decoder_dim, masking_ratio=self.encoder_masking_ratio, decoder_depth=1)

        elif self.finetune1:

            self.encoder.requires_grad_(False)
            self.encoder.eval()

            #self.hrrr_decoder = ERAdecoder(final_image_size=(670, 260),
            #    patch_size=(10,10), 
            #    final_channels=decoder_channels,
            #    dim=192, #decoder_dim,
            #    depth=decoder_depth,
            #    heads=decoder_heads,
            #    encoder_dim=encoder_dim,
            #    mlp_dim=decoder_mlp_dim) 

            #self.hrrr_decoder = self.hrrr_decoder.to(self.accelerator.device)

            #self.model = MAE_decoder(encoder=self.encoder, hrrr_decoder = self.hrrr_decoder, decoder_dim=192,
            self.model = MAE_decoder(encoder=self.encoder, decoder_dim=192,
                                     encoder_masking_ratio=self.encoder_masking_ratio, decoder_depth=decoder_depth,
                                     decoder_masking_ratio=self.decoder_masking_ratio)


            self.model.encoder.requires_grad_(False)

            for layer in range(4,8):
                for param in self.model.encoder.transformer.layers[layer].parameters():
                    param.requires_grad = True

            """ these have been commented out recently """
            #for param in self.model.hrrr_decoder.enc_to_dec.parameters():
            #    param.requires_grad = True

            #for param in self.model.hrrr_decoder.enc_to_dec_pos.parameters():
            #    param.requires_grad = True

            #for param in self.model.hrrr_decoder.transformer.parameters():
            #    param.requires_grad = False #!!! JRD JRD JRD

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)

        elif self.finetune2:

            self.encoder.requires_grad_(False)
            self.encoder.eval()

            self.hrrr_decoder = ERAdecoder(final_image_size=(670, 260),
                patch_size=(10,10), 
                final_channels=decoder_channels,
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                encoder_dim=encoder_dim,
                mlp_dim=decoder_mlp_dim) 

            self.hrrr_decoder = self.hrrr_decoder.to(self.accelerator.device)


            self.hrrr_decoder.requires_grad_(False)
            self.hrrr_decoder.eval()

            """ This is for the finetuning of the conv head """
            self.model = ERA5Upscaler(self.encoder, self.hrrr_decoder)

            self.model.encoder.requires_grad_(False)

            """ this line was added for final finetuning """
            self.model.hrrr_decoder.requires_grad_(False)

            """ this de-thaw was used for finetuning training """
            for layer in range(0,4): # was 2,4
                for param in self.model.hrrr_decoder.transformer.layers[layer].parameters():
                    param.requires_grad = True

            """ these have been commented out recently """
            for param in self.model.hrrr_decoder.enc_to_dec.parameters():
                param.requires_grad = True

            for param in self.model.hrrr_decoder.enc_to_dec_pos.parameters():
                param.requires_grad = True

            #for name, param in self.model.named_parameters():
            #    if param.requires_grad:
            #        print(name)


        num_params = count_params(self.model)
        print(f"Number of parameters: {num_params:,}")
        self.model = self.model.to(self.accelerator.device)

        self.train_dataset = dataloader_superres( \
                    era5_path,
                    hrrr_path,
                    train_file,
                    pretraining = self.pretrain,
                    use_npy=self.npy,
                    two_hours=False)

        self.patch_mask_indices = torch.tensor(self.train_dataset.patch_mask_indices)
        self.patch_mask_anti_indices = torch.tensor(self.train_dataset.patch_mask_anti_indices)

        self.val_dataset = dataloader_superres( \
                    era5_path,
                    hrrr_path,
                    val_file,
                    pretraining = self.pretrain,
                    use_npy=self.npy,
                    two_hours=False)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=4)

        if self.accelerator.is_main_process:

            self.val_dataloader = DataLoader(self.val_dataset,
                                               shuffle=True,#not self.validate,
                                               batch_size=self.batch_size,
                                               num_workers=2)
        else:

            self.val_dataloader = DataLoader(self.val_dataset,
                                               shuffle=True,# not self.validate,
                                               batch_size=self.batch_size,
                                               num_workers=2)

        self.decoder_learning_rate = decoder_learning_rate
        self.encoder_learning_rate = encoder_learning_rate

        if self.load_checkpoint_bool and self.legacy_checkpoint_bool:
            self.load_checkpoint_legacy(checkpoint_path)

        if self.pretrain:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=encoder_learning_rate, betas=(self.beta1,self.beta2), weight_decay=encoder_weight_decay, eps=self.eps)

        else:
            """ For 2nd MAE """
            self.optimizer = torch.optim.AdamW([\
            {'params': [param for name, param in self.model.named_parameters() if any([f'encoder.transformer.layers.{n}' in name for n in range(4,8)])], 'lr':encoder_learning_rate, 'weight_decay':encoder_weight_decay},
            {'params': [param for name, param in self.model.named_parameters() if not "encoder" in name and not "patch" in name], 'lr':decoder_learning_rate, 'weight_decay':decoder_weight_decay}],betas=(self.beta1,self.beta2), eps=self.eps)
            #""" For Fine tuning """
            #self.optimizer = torch.optim.AdamW([\
            #{'params': [param for name, param in self.model.named_parameters() if any([f'encoder.transformer.layers.{n}' in name for n in range(4,8)])], 'lr':encoder_learning_rate, 'weight_decay':encoder_weight_decay},
            #{'params': [param for name, param in self.model.named_parameters() if any([f'hrrr_decoder.transformer.layers.{n}' in name for n in range(0,4)])], 'lr':encoder_learning_rate, 'weight_decay':encoder_weight_decay},
            #                                    {'params': [param for name, param in self.model.named_parameters() if not "hrrr_decoder" in name and not "encoder" in name and not "patch" in name],
            #                                     'lr':decoder_learning_rate, 'weight_decay':decoder_weight_decay}],
            #                                    betas=(0.9,0.999), eps=1e-8)
            


        ##self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=24, eta_min=1e-7) 
        ##self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma=0.94)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 100*self.accelerator.num_processes, gamma=0.977)
        ##self.scheduler1 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.977)

        if self.pretrain:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.accelerator.num_processes, gamma=0.977) #was 9877, 0.977
            #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,T_0=1000,eta_min=0.05*self.decoder_learning_rate,T_mult=2)

        elif False:
            self.scheduler1 = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.accelerator.num_processes, gamma=0.8914) #0.965)
            self.scheduler2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,T_0=100,eta_min=0.1*self.decoder_learning_rate,T_mult=2)
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.scheduler1,self.scheduler2],milestones=[400])

            self.scheduler1 = self.accelerator.prepare(self.scheduler1)
            self.scheduler2 = self.accelerator.prepare(self.scheduler2)
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 8*self.accelerator.num_processes, gamma=0.977) #was 0.977
            #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,T_0=1000,eta_min=0.05*self.decoder_learning_rate,T_mult=2)

        self.val_dataloader, self.train_dataloader, self.model, self.optimizer = self.accelerator.prepare(
                self.val_dataloader, self.train_dataloader, self.model, self.optimizer)

        self.scheduler = self.accelerator.prepare(self.scheduler)


        if self.load_checkpoint_bool and not self.legacy_checkpoint_bool:

            self.accelerator.load_state(checkpoint_path)

        if self.accelerator.is_main_process:

            logging.info(self.accelerator.state)

        """
        self.model = self.accelerator.unwrap_model(self.model)
        state_dict = self.model.state_dict()
        save_dict = {}
        save_dict['model']={'state_dict':state_dict}
        #torch.save(save_dict, "/pscratch/sd/j/jderm/july19_14ch_epoch40.pt")
        torch.save(save_dict, "/pscratch/sd/j/jderm/july19_14ch_finetune.pt")
        sys.exit(0)
        """

        

    def train(self):
        logging.debug("starting training")

        if self.checkpoint_epoch is None:
            start_epoch = 0
        else:
            start_epoch = self.checkpoint_epoch + 1

        start_epoch=0

        for self.i_epoch in range(start_epoch, self.n_epochs):

            logging.debug(f"starting epoch {self.i_epoch}")

            #self.decoder_mask_prob = 0.5 - 0.5*np.cos(np.pi*((self.i_epoch+1)/100)) # was 200, /4
            #self.decoder_mask_prob = 1 - np.cos(np.pi*((self.i_epoch+1)/200)) # was 200, /4
            self.decoder_mask_prob = 0.5 - 0.5 * np.cos(np.pi*((self.i_epoch+1)/200)) # was 200, /4
            self.decoder_mask_prob = min(1,self.decoder_mask_prob)

            if self.i_epoch >= 200:
                self.decoder_mask_prob = 1.0

            #self.decoder_mask_prob = 0.995

            logging.debug(f"{self.decoder_mask_prob=}")

            self.one_epoch()

            os.system("nvidia-smi > ~/nvidia.log")

            #self.scheduler.step()

            if self.i_epoch > 0 and self.i_epoch % self.save_interval == 0:

                self.accelerator.wait_for_everyone()

                self.save_model()

    def load_checkpoint_legacy(self,checkpoint_path):

        if not os.path.exists(checkpoint_path):
            logging.error(f"weight file {checkpoint_path} does not exist")
            sys.exit(0)

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        model_key = list(checkpoint.keys())[0]

        contains_decoder_hrrr = any(["hrrr_decoder" in key for key in checkpoint[model_key].keys()])
        contains_final_decoder_hrrr = any(["hrrr_decoder_final" in key for key in checkpoint[model_key].keys()])

        try:
            self.model.load_state_dict(checkpoint[model_key]['state_dict'])
            return 
    
        except:
            logging.debug("Re-loading checkpoint: key mismatch")

        if self.pretrain:

            reduced_keys = [key for key in checkpoint[model_key]['model_state_dict'].keys() if not "mlp" in key]

            self.model.load_state_dict({key: checkpoint[model_key]['model_state_dict'][key] for key in reduced_keys})
            
            self.i_step  = checkpoint[model_key]['step']

            self.checkpoint_epoch = checkpoint[model_key]['epoch']

            #self.optimizer.load_state_dict(checkpoint[model_key]['optimizer_state_dict'])

        elif self.finetune1:
            self.first_finetune_epoch = True
            """ checkpoint does not contain decoder_hrrr. It is therefore a checkpoint from pretraining.
                No epochs, steps, or optimizer values are loaded
            """

            encoder_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "encoder." in key and not "mlp" in key]
            self.model.encoder.load_state_dict({key.removeprefix('encoder.'): checkpoint[model_key]['state_dict'][key] for key in encoder_keys})
            # JRD WHY IS THTIS HERE
            #enc2dec_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "enc_to_dec_pos." in key]
            #self.model.hrrr_decoder.enc_to_dec_pos.load_state_dict({key.removeprefix('enc_to_dec_pos.'): checkpoint[model_key]['state_dict'][key] for key in enc2dec_keys})

            #enc2dec_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "enc_to_dec." in key]
            #self.model.hrrr_decoder.enc_to_dec.load_state_dict({key.removeprefix('enc_to_dec.'): checkpoint[model_key]['state_dict'][key] for key in enc2dec_keys})

        elif self.finetune2:

            if True and self.accelerator.is_main_process:
                logging.debug(self.model.state_dict().keys())

            if True and self.accelerator.is_main_process:
                logging.debug(checkpoint[model_key]['state_dict'].keys())


            encoder_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "encoder." in key and not "mlp" in key]
            self.model.encoder.load_state_dict({key.removeprefix('encoder.'): checkpoint[model_key]['state_dict'][key] for key in encoder_keys})
            
            decoder_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "hrrr_decoder." in key and not "mlp" in key]
            self.model.hrrr_decoder.load_state_dict({key.removeprefix('hrrr_decoder.'): checkpoint[model_key]['state_dict'][key] for key in decoder_keys})

            enc2dec_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "enc_to_dec_pos." in key and "hrrr" not in key]
            self.model.enc_to_dec_pos.load_state_dict({key.removeprefix('enc_to_dec_pos.'): checkpoint[model_key]['state_dict'][key] for key in enc2dec_keys})

            enc2dec_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "enc_to_dec." in key and "hrrr" not in key]
            self.model.enc_to_dec.load_state_dict({key.removeprefix('enc_to_dec.'): checkpoint[model_key]['state_dict'][key] for key in enc2dec_keys})

            token_key = [key for key in checkpoint[model_key]['state_dict'].keys() if "token" in key and "encoder" not in key and "hrrr_decoder" not in key]
            self.model.t_token.data = checkpoint[model_key]['state_dict']["t_token"]
            self.model.tminus1_token.data = checkpoint[model_key]['state_dict']["tminus1_token"]


        else:
            """ checkpoint contains both encoder and decoder_hrrr, 
                suggesting that it is a checkpoint after fine-tuning has commenced
            """
            logging.debug("final checkpoint load point")

            decoder_keys = [key for key in checkpoint[model_key]['model_state_dict'].keys() if "decoder_hrrr." in key]
            
            self.model.decoder_hrrr.load_state_dict({key.removeprefix('decoder_hrrr.'): checkpoint[model_key]['model_state_dict'][key] for key in decoder_keys})

            self.i_step  = checkpoint[model_key]['step']

            self.checkpoint_epoch = checkpoint[model_key]['epoch']

            self.optimizer.load_state_dict(checkpoint[model_key]['optimizer_state_dict_decoder'])

        del checkpoint

        torch.cuda.empty_cache()

        logging.debug("model load successful")

    def save_model(self):

        current_time = datetime.now()

        checkpoint_path  = f'/pscratch/sd/j/jderm/runs/{self.experiment_string}_'
        checkpoint_path += f'epoch{self.i_epoch}/'

        self.accelerator.save_state(output_dir = checkpoint_path, safe_serialization = False)

        logging.debug(f"model saved: {checkpoint_path}")


    def one_epoch(self):

        root = logging.getLogger()  # root logger
        for h in root.handlers:
            h.flush()

        for self.i_batch, (sample_x, sample_y, idx) in enumerate(self.train_dataloader):

            # 14150 has been added for finetuning experiments
            self.accumulated_batches = (self.i_step) // self.grad_accumulation_steps

            if self.warmup and self.accumulated_batches <= self.warmup_steps:

                for i,g in enumerate(self.optimizer.param_groups):

                    g['lr'] = np.power(10,5*(self.accumulated_batches/self.warmup_steps) - 5) * [self.encoder_learning_rate, self.decoder_learning_rate][i]

            if not self.validate:

                if self.pretrain:

                    with self.accelerator.accumulate(self.model):

                        self.one_batch_pretrain(sample_x, sample_y, phase="train")

                else:

                    with self.accelerator.accumulate(self.model):

                        self.one_batch_finetune(sample_x, sample_y, phase="train")

                if self.accelerator.is_main_process and self.i_step % self.val_log_interval == 0:

                    for i,g in enumerate(self.optimizer.param_groups):

                        self.accelerator.log({f"lr-{i}":g['lr']}, step=self.i_step // self.grad_accumulation_steps)
                        self.accelerator.log({f"norm-lr-{i}":g['lr'] / (self.accelerator.num_processes * self.grad_accumulation_steps)}, step=self.i_step // self.grad_accumulation_steps)

                if self.accelerator.is_main_process and self.accumulated_batches % self.train_log_interval == 0 and  self.i_step % self.grad_accumulation_steps == self.grad_accumulation_steps - 1:

                    if self.total_norm is not None:
                        self.accelerator.log({"grad_norm":self.total_norm.mean().item()}, step=self.i_step // self.grad_accumulation_steps)
                        self.accelerator.log({"grad_norm_steps":self.total_norm.mean().item()}, step=self.i_step)

                if self.accelerator.is_local_main_process and self.i_step % self.gpu_interval == 0:
                    for i in range(0,4):
                        self.accelerator.log({f"process: {self.accelerator.process_index} gpu:{i}":torch.cuda.utilization(i)}, step=self.i_step // self.grad_accumulation_steps)

                if self.i_step % self.val_log_interval == 0:
                    self.eval("val-eval", reconstruct=False)

                if self.i_step % self.train_log_interval == 0:
                    self.eval("train-eval", reconstruct=False)

                if self.accelerator.is_main_process and self.i_step % self.train_log_interval == 0:
                    logging.info(f"TRAIN loss {self.global_train_loss.detach().mean().item()}")
                    self.accelerator.log({"masking_prob":self.decoder_mask_prob}, step=self.i_step // self.grad_accumulation_steps)
                    self.accelerator.log({"train_loss":self.global_train_loss.detach().mean().item()}, step=self.i_step // self.grad_accumulation_steps)
                    self.accelerator.log({"train_loss_steps":self.global_train_loss.detach().mean().item()}, step=self.i_step)

                if self.accelerator.is_main_process and self.i_step % self.val_log_interval == 0:
                    logging.info(f"VAL loss {self.global_val_loss.detach().mean().item()}")
                    self.accelerator.log({"val_loss":self.global_val_loss.detach().mean().item()}, step=self.i_step // self.grad_accumulation_steps)
                    self.accelerator.log({"val_loss_steps":self.global_val_loss.detach().mean().item()}, step=self.i_step)
                self.i_step += 1


            else:
                self.global_val_loss = torch.tensor([], device=self.device)
                self.eval("val-eval", reconstruct=False)
                logging.info(f"VAL loss {self.global_val_loss.detach().mean().item()}")
                self.i_step += 1
                break


    def eval(self, phase,reconstruct=False):

        self.model.eval()


        if phase == "val-eval" or phase == "val":
            dataset = self.val_dataloader

        elif phase == "train-eval":
            dataset = self.train_dataloader

        for j, batch in enumerate(dataset):

            with torch.no_grad():

                sample_x, sample_y, idx = batch

                if self.pretrain:
                    self.one_batch_pretrain(sample_x, sample_y, phase=phase, reconstruct=reconstruct)
                else:
                    self.one_batch_finetune(sample_x, sample_y, phase=phase, reconstruct=reconstruct, index=idx)



                #if j % 10 == 0:
                #    logging.info(f"VAL loss {self.global_val_loss.detach().mean().item()}")
                #    logging.debug(f"Validated on {j} data")

            if not self.validate:
                break

        self.model.train()


    def one_batch_finetune(self, sample_x, sample_y, phase="train", reconstruct=False, index=None):

        test1 = None
        test2 = None
        if self.finetune1:
            reconstruction, sample_y2,_, test1, test2 = self.model(sample_x, sample_y, two_hours=False, decoder_masking_ratio=self.decoder_mask_prob)


        if self.finetune2:
            reconstruction, sample_y2, y_full = self.model(sample_x, sample_y, self.patch_mask_indices, self.patch_mask_anti_indices)

        if False:
            #y_full = y_full.detach().cpu().numpy()

            np.save(f"/pscratch/sd/j/jderm/results_aug_4day_recon/recon_{self.i_step}_{self.accelerator.process_index}.npy", reconstruction.detach().cpu().numpy())
            np.save(f"/pscratch/sd/j/jderm/results_aug_4day_recon/index_{self.i_step}_{self.accelerator.process_index}.npy", index.cpu().numpy())

            #with open(f"/pscratch/sd/j/jderm/results_aug/ft5_final_{self.i_step}_{self.accelerator.process_index}.pkl","wb") as f:
            #    pickle.dump(list(file_names.cpu().numpy()),f)

        if phase == "train":
            loss = F.mse_loss(reconstruction, sample_y2,reduction='none')
        else:
            loss = F.mse_loss(reconstruction, sample_y2,reduction='none')
        loss = loss.reshape(sample_x.shape[0], -1)
        # CONSISTENCY
        loss = torch.mean(loss, dim=1)
        if test1 is not None and test2 is not None:
            loss += F.l1_loss(test1, test2)

        if torch.isnan(loss.mean()):
            logging.info(f"Nan in loss!")
            return

        if phase == "train":
            loss = torch.sum(loss)
            self.accelerator.backward(loss)
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm, norm_type=2)
            
            if False and self.accelerator.is_main_process:
                logging.info(f"TRAIN-debug loss {loss.detach().mean().item()}")

            self.optimizer.step()

            if self.i_step % 100 == 0: #JRD JULY
                self.scheduler.step()

            if self.accumulated_batches % self.train_log_interval and  self.i_step % self.grad_accumulation_steps == self.grad_accumulation_steps - 1:

                total_norm = 0
                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]

                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5
                self.total_norm = self.accelerator.gather_for_metrics(torch.tensor(total_norm,device=self.device))

            self.optimizer.zero_grad()
        
        elif phase == "val-eval" and reconstruct and False:

            loss = torch.mean(loss)
            self.global_val_loss = self.accelerator.gather_for_metrics(loss)

            ag = integrated_gradients_for_spatial_output(self.accelerator.unwrap_model(self.model), sample_x[0].unsqueeze(0), target_h=(669-191)//10,target_w=143//10, target_dim=None) #Morro
            ##ag = integrated_gradients_for_spatial_output(self.accelerator.unwrap_model(self.model), sample_x[0].unsqueeze(0), target_h=(669-401)//10,target_w=120//10, target_dim=None) #Humboldt

            arr = ag[0].detach().cpu().numpy()

            
            np.save(f"/pscratch/sd/j/jderm/results_aug_2day_recon/morro_index_{self.i_step}_{self.accelerator.process_index}.npy", index.cpu().numpy())
            #np.save(f"/pscratch/sd/j/jderm/results_aug_2day_recon/morro_results_{self.i_step}_{self.accelerator.process_index}.npy", arr)
            self.i_step += 1

            if False:
                with torch.no_grad():
                    patches, masked_indices, pred_pixel_values = self.accelerator.unwrap_model(self.model).reconstruct(sample_x[:].to(self.device), 
                                                                                                                       sample_y[:].to(self.device),
                                                                                                                       decoder_masking_ratio=1.0,
                                                                                                                       two_hours=False)


                b_, p_, dim_ = patches.shape
                b_range = torch.arange(0,b_)[:,None].to(self.device)

                hrrr_all_indices = torch.arange(1742).unsqueeze(0).repeat(b_,1).to(self.device)

                hrrr_patch_mask = torch.tensor(self.train_dataset.patch_mask).unsqueeze(0).repeat(b_,1).to(self.device)

                hrrr_outofbound = hrrr_all_indices[:,~hrrr_patch_mask[0]].reshape(b_,-1)
                hrrr_withinbound = hrrr_all_indices[:,hrrr_patch_mask[0]].reshape(b_,-1)

                reconstructed_img = torch.zeros(b_,1742,dim_).to(self.device)
                reconstructed_img[b_range,hrrr_outofbound]  = torch.zeros(dim_).to(self.device)
                reconstructed_img[b_range,hrrr_withinbound] = pred_pixel_values[b_range,masked_indices.argsort(dim=1)]
                reconstructed_img = einops.rearrange(reconstructed_img, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=67, w=26,  p1=10, p2=10, c=2) 
                

                ch0_mean = hrrr_mean_var[0][0]
                ch0_var = hrrr_mean_var[0][1]
                ch1_mean = hrrr_mean_var[1][0]
                ch1_var = hrrr_mean_var[1][1]

                reconstructed_img[:,0] *= ch0_var 
                reconstructed_img[:,0] += ch0_mean
                reconstructed_img[:,1] *= ch1_var 
                reconstructed_img[:,1] += ch1_mean

                np.save(f"/pscratch/sd/j/jderm/results_aug_1day_recon/recon_{self.i_step}_{self.accelerator.process_index}.npy", reconstructed_img.detach().cpu().numpy())
                np.save(f"/pscratch/sd/j/jderm/results_aug_1day_recon/index_{self.i_step}_{self.accelerator.process_index}.npy", index.cpu().numpy())
                np.save(f"/pscratch/sd/j/jderm/results_aug_1day_recon/gt_{self.i_step}_{self.accelerator.process_index}.npy", sample_y.cpu().numpy())
                self.i_step += 1

            """
            new_patches = torch.zeros(b_, 1742, dim_).to(self.device)
 
            patch_mask = torch.tensor(self.train_dataset.patch_mask).to(self.device)
            new_patches[:,patch_mask] = patches
 
            _, n_masked = masked_indices.shape
            new_pixel_values = torch.zeros(b_, n_masked, 200).to(self.device)

            #new_masked_indices = torch.arange(0,1742).int().reshape(1,-1).to(self.device)
            new_masked_indices = torch.arange(0,1742).int().reshape(1,-1).repeat(b_,1).to(self.device)
            new_masked_indices = new_masked_indices[:,patch_mask][:,masked_indices]

            sample_y_tmp = torch.zeros(b_, 1742, 200).to(self.device)
            #sample_y_tmp[:,patch_mask] = sample_y[:1]
            sample_y_tmp[:,patch_mask] = sample_y[:]
            sample_y = sample_y_tmp
            sample_y = einops.rearrange(sample_y, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=67, w=26,  p1=10, p2=10, c=2) 

            img = reconstruct_image(new_patches, sample_y, masked_indices=new_masked_indices, pred_pixel_values=pred_pixel_values, patch_size=10)
            imgMasked = reconstruct_image(new_patches, sample_y, masked_indices=new_masked_indices, patch_size=10)

            sample_y = sample_y.detach().cpu().numpy()
            img = img.detach().numpy()

            B,W,H,C = img.shape

            new_image_ch0 = np.zeros(shape=(1,1,H,3*W))
            new_image_ch0[0,0,:H,0*W:1*W] = sample_y[0,0]
            new_image_ch0[0,0,:H,1*W:2*W] = imgMasked[0,:,:,0].T
            new_image_ch0[0,0,:H,2*W:3*W] = img[0,:,:,0].T

            """
            if self.accelerator.is_main_process and False:
                self.accelerator.get_tracker("tensorboard").log_images({"u-200": new_image_ch0}, step=self.i_step)

            if False:
                logging.info("here i am")

                loss = loss.mean()

                """
                DEBUG:root:torch.Size([910, 200])
                DEBUG:root:torch.Size([910, 200])
                """

                ch0_mean = hrrr_mean_var[0][0]
                ch0_var = hrrr_mean_var[0][1]
                ch1_mean = hrrr_mean_var[1][0]
                ch1_var = hrrr_mean_var[1][1]

                sample_y = einops.rearrange(sample_y,'b hw (p1 p2 c) -> b hw c (p1 p2)', c=2, p1=10, p2=10).cpu()
                reconstruction = einops.rearrange(reconstruction,'b hw (p1 p2 c) -> b hw c (p1 p2)', c=2, p1=10, p2=10).cpu()

                sample_y_prime = torch.zeros_like(sample_y)
                sample_y_prime[:,:,0] = sample_y[:,:,0]*ch0_var + ch0_mean
                sample_y_prime[:,:,1] = sample_y[:,:,1]*ch1_var + ch1_mean

                reconstruction_prime = torch.zeros_like(reconstruction)
                reconstruction_prime[:,:,0] = reconstruction[:,:,0]*ch0_var + ch0_mean
                reconstruction_prime[:,:,1] = reconstruction[:,:,1]*ch1_var + ch1_mean

                """
                DEBUG:root:torch.Size([1, 910, 2, 100])
                """

                sample_y = einops.rearrange(sample_y_prime,'b hw c (p1 p2) -> b hw (p1 p2 c)' , c=2, p1=10, p2=10)
                reconstruction = einops.rearrange(reconstruction_prime,'b hw c (p1 p2) -> b hw (p1 p2 c)' , c=2, p1=10, p2=10)

                for i in range(0,self.batch_size):
                    gt, pred = reconstruct_image_reduced(self.train_dataset.patch_mask_indices, sample_y[i], reconstruction[i])

                    gt =  gt.detach().cpu().numpy()
                    pred = pred.detach().cpu().numpy()

                    """
                    DEBUG:root:(2, 670, 260)
                    DEBUG:root:(2, 670, 260)
                    """
                    C,H,W = gt.shape

                    """
                    new_image_ch0 = np.zeros(shape=(1, 1, H, 2*W))
                    new_image_ch1 = np.zeros(shape=(1, 1, H, 2*W))

                    new_image_ch0[:,:,:H,0:W] = gt[0]
                    new_image_ch0[:,:,:H,W:2*W] = pred[0]

                    new_image_ch1[:,:,:H,0:W] = gt[1]
                    new_image_ch1[:,:,:H,W:2*W] = pred[1]

                    """
                    new_image_ch0 = np.zeros(shape=(1, 2, H, W))
                    new_image_ch1 = np.zeros(shape=(1, 2, H, W))

                    new_image_ch0[:,0,:H,:W] = gt[0]
                    new_image_ch0[:,1,:H,:W] = pred[0]

                    new_image_ch1[:,0,:H,:W] = gt[1]
                    new_image_ch1[:,1,:H,:W] = pred[1]


                    self.u80_list.append(new_image_ch0)
                    self.v80_list.append(new_image_ch1)

                    logging.debug(f"images: {len(self.u80_list)}")
                    
                    if len(self.u80_list) == 24 and self.accelerator.is_main_process:

                        u80 = np.vstack(self.u80_list)
                        v80 = np.vstack(self.v80_list)

                        np.save(f"/pscratch/sd/j/jderm/results_v5.0/u_results_{self.uv_iter}_V3.npy", u80)
                        np.save(f"/pscratch/sd/j/jderm/results_v5.0/v_results_{self.uv_iter}_V3.npy", v80)

                        self.uv_iter += 1
                        logging.info(f"{self.uv_iter}")

                        self.u80_list = list()
                        self.v80_list = list()

            #tracker=self.accelerator.get_tracker("tensorboard")

            #tracker.log_images({"u-80": new_image_ch0}, step=self.i_step)
            #tracker.log_images({"v-80": new_image_ch1}, step=self.i_step)

        elif phase=="val-eval":
            print(f'{loss.shape=}')
            #loss = torch.sum(loss).mean() #JRD why was this sum??
            loss = torch.mean(loss)
            #self.global_val_loss = torch.cat([self.global_val_loss, self.accelerator.gather_for_metrics(loss)])
            self.global_val_loss = self.accelerator.gather_for_metrics(loss)

        elif phase=="train-eval":
            loss = torch.mean(loss)
            self.global_train_loss = self.accelerator.gather_for_metrics(loss)



    def one_batch_pretrain(self, sample_x, sample_y, phase="train", reconstruct=False):

        if phase == "train":

            self.reset_grad()

        if phase == "train":

            pred_pixel_values, masked_pixels,_ = self.model(sample_x)

            loss = F.mse_loss(pred_pixel_values, masked_pixels, reduction='none')
            loss = loss.reshape(sample_x.shape[0], -1)
            loss = torch.mean(loss, dim=1)
            loss = torch.sum(loss)

            self.accelerator.backward(loss)

            self.optimizer.step()

            if self.i_step % 100 == 0 and self.accumulated_batches > self.warmup_steps:
                self.scheduler.step()

            if self.accumulated_batches % self.train_log_interval == 0 and  self.i_step % self.grad_accumulation_steps == self.grad_accumulation_steps - 1:

                self.global_loss = self.accelerator.gather_for_metrics(loss/sample_x.shape[0])

                total_norm = 0

                parameters = [p for p in self.model.parameters() if p.grad is not None and p.requires_grad]

                for p in parameters:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2

                total_norm = total_norm ** 0.5

                self.total_norm = self.accelerator.gather_for_metrics(torch.tensor(total_norm,device=self.device))

        else:

            pred_pixel_values, masked_pixels,_ = self.model(sample_x, masking_ratio = self.eval_masking_ratio, masking_strategy = self.eval_masking_strategy)

            loss = F.mse_loss(pred_pixel_values, masked_pixels, reduction='mean')

        if phase == "val-eval":

            self.global_val_loss = self.accelerator.gather_for_metrics(loss)

            if self.accelerator.is_main_process and reconstruct:

                patches, masked_indices, pred_pixel_values = self.accelerator.unwrap_model(self.model).reconstruct(sample_x[:1].to(self.device))

                img = reconstruct_image(patches, sample_x, masked_indices=masked_indices, pred_pixel_values=pred_pixel_values, patch_size=10)
                imgMasked = reconstruct_image(patches, sample_x, masked_indices=masked_indices, patch_size=10)

                sample_x = sample_x.detach().cpu().numpy()
                img = img.detach().numpy()

                B,W,H,C = img.shape

                new_image_ch0 = np.zeros(shape=(1,15,H,3*W))
                for ch_ in range(0,15):
                    new_image_ch0[0,ch_,:H,0*W:1*W] = sample_x[0,ch_]
                    new_image_ch0[0,ch_,:H,1*W:2*W] = imgMasked[0,:,:,ch_].T
                    new_image_ch0[0,ch_,:H,2*W:3*W] = img[0,:,:,ch_].T

                #np.save(f"/pscratch/sd/j/jderm/results_v3.0/mae_viz_{self.i_step}.npy",new_image_ch0)

                #tracker=self.accelerator.get_tracker("tensorboard")
                #tracker.log_images({"u-200": new_image_ch0}, step=self.i_step)


            self.global_val_loss = self.accelerator.gather_for_metrics(loss)

        elif phase=="train-eval":

            self.global_train_loss = self.accelerator.gather_for_metrics(loss)


    def reset_grad(self):

        self.optimizer.zero_grad()

def parse_arg():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--checkpoint-path', type=str, required=False)
    parser.add_argument('--experiment-string', type=str, required=False)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--era5-path', type=str, required=True)
    parser.add_argument('--hrrr-path', type=str, required=True)
    parser.add_argument('--val-file', type=str, required=True)
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--batch-size', type=int, required=False)
    parser.add_argument('--grad-accumulation-steps', type=int, required=False)
    parser.add_argument('--epochs',type=int,required=False)
    parser.add_argument('--pretrain',action="store_true")
    parser.add_argument('--finetune1',action="store_true")
    parser.add_argument('--finetune2',action="store_true")
    parser.add_argument('--validate',action="store_true")
    parser.add_argument('--npy',action="store_true")
    parser.add_argument('--dataparallel',action="store_true")
    parser.add_argument('--seed',type=int, required=False)

    parser.add_argument('--encoder-learning-rate',type=float,required=False)
    parser.add_argument('--decoder-learning-rate',type=float,required=False)
    parser.add_argument('--encoder-weight-decay',type=float,required=False)
    parser.add_argument('--decoder-weight_decay',type=float,required=False)
    parser.add_argument('--grad-norm',type=float,required=False)
    parser.add_argument('--beta1',type=float,required=False)
    parser.add_argument('--beta2',type=float,required=False)
    parser.add_argument('--eps',type=float,required=False)

    parser.add_argument('--encoder-depth',type=int,required=False)
    parser.add_argument('--encoder-dim',type=int,required=False)
    parser.add_argument('--encoder-channels',type=int,required=False)
    parser.add_argument('--encoder-heads',type=int,required=False)
    parser.add_argument('--encoder-mlp_dim',type=int,required=False)
    parser.add_argument('--encoder-num-registers',type=int,required=False)
    parser.add_argument('--encoder-masking-ratio',type=float,required=False)


    parser.add_argument('--decoder-depth',type=int,required=False)
    parser.add_argument('--decoder-dim',type=int,required=False)
    parser.add_argument('--decoder-channels',type=int,required=False)
    parser.add_argument('--decoder-heads',type=int,required=False)
    parser.add_argument('--decoder-mlp_dim',type=int,required=False)
    parser.add_argument('--decoder-masking-ratio',type=float,required=False)

    args = parser.parse_args()

    return args

def main():
    args = parse_arg()

    trainingparam = TrainingParam(**{str(field.name):vars(args)[field.name] for field in fields(TrainingParam) if field.name in vars(args).keys()})
    modelparam = ModelParam(**{str(field.name):vars(args)[field.name] for field in fields(ModelParam) if field.name in vars(args).keys()})
    optparam = OptParam(**{str(field.name):vars(args)[field.name] for field in fields(OptParam) if field.name in vars(args).keys()})


    logging.debug(trainingparam)
    logging.debug(modelparam)
    logging.debug(optparam)

    obj = trainer(**vars(trainingparam),**vars(modelparam), **vars(optparam))
    obj.train()

if __name__ == '__main__':
    main()

from accelerate import Accelerator
