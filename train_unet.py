from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
import argparse
import os, sys
import os.path as path
from datetime import datetime
import logging
from data.dataloader_nc import dataloader_superres
from models.models import  ERAencoder, ERAdecoder, ERA5Upscaler
from models.mae import MAE
from models.unet import UNet
from utils.forecast_metrics import reconstruct_image_reduced, plot_reconstruction, reconstruct_image
from utils.arguments import TrainingParam, ModelParam, OptParam
from torch.utils.data import DataLoader
from itertools import cycle
import einops
import numpy as np
logging.basicConfig(level=logging.DEBUG)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass, fields
import pickle

hrrr_mean_var = np.array([ \
[-1.5728172063827515,  3.915628433227539],
[-1.3613358736038208,  4.049429893493652]])


# Count the total number of parameters
def count_params(model):
    return sum(p.numel() for p in model.parameters())


#unet why are these so differnet
hrrr_mean_var = np.array([ \
 [ 1.34187811,  3.89167407],
 [-2.75422505,  6.09448469]])

class trainer():
    def __init__(self, device, checkpoint_path, experiment_string, pretrain, finetune1, finetune2, npy ,epochs, batch_size, dataparallel,
                 era5_path, hrrr_path, train_file, val_file, grad_accumulation_steps, validate,
                 encoder_learning_rate, encoder_weight_decay, decoder_learning_rate, decoder_weight_decay,
                 encoder_depth, encoder_dim, encoder_channels, encoder_heads, encoder_mlp_dim, encoder_num_registers, 
                 decoder_depth, decoder_dim, decoder_channels, decoder_heads, decoder_mlp_dim,
                 encoder_masking_ratio, decoder_masking_ratio, grad_norm, seed, beta1, beta2, eps):


        self.accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps, 
                                       log_with="tensorboard",
                                       project_dir=f"/pscratch/sd/j/jderm/tb_logdir_unet/{experiment_string}",
                                       step_scheduler_with_optimizer=False)
        self.accelerator.init_trackers(experiment_string)


        set_seed(self.accelerator.process_index)

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

        self.pretrain = pretrain
        self.npy = npy
        self.warmup = True
        self.warmup_steps = 500
        self.first_finetune_epoch = False
        self.dataparallel = dataparallel



        self.batch_size = batch_size
        self.n_epochs = epochs
        self.experiment_string = experiment_string
        self.train_log_interval = 10

        self.save_interval = 1
        self.val_interval = self.train_log_interval*2

        self.i_epoch = 0
        self.i_batch = 0
        self.i_step = 0
        self.global_loss = None


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

        used, total = torch.cuda.mem_get_info(self.device)
        if total > 60000000000:
            self.batch_size = int(1.3*self.batch_size)
            encoder_learning_rate *= 1.5
            decoder_learning_rate *= 1.5
        else:
            self.batch_size *= 1
            encoder_learning_rate *= 1
            decoder_learning_rate *= 1

        self.model = UNet(15,2)
        num_params = count_params(self.model)
        self.model = self.model.to(self.accelerator.device)

        self.hrrr_mask = np.load("/global/common/software/m4506/s2s_mae/training/s2s_tar/hrrr_mask_reduced.npy")
        z = einops.rearrange(self.hrrr_mask,'(h p1) (w p2) -> (h w) (p1 p2)', p1 = 10, p2 = 10)
        self.patch_mask = np.sum(z, axis=1) > 0

        self.train_dataset = dataloader_superres( \
                    era5_path,
                    hrrr_path,
                    train_file,
                    pretraining = self.pretrain,
                    use_npy=self.npy)

        self.val_dataset = dataloader_superres( \
                    era5_path,
                    hrrr_path,
                    val_file,
                    pretraining = self.pretrain,
                    use_npy=self.npy)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           pin_memory=True,
                                           num_workers=4)

        if self.accelerator.is_main_process:

            self.val_dataloader = DataLoader(self.val_dataset,
                                               shuffle=False,
                                               pin_memory=True,
                                               batch_size=self.batch_size,
                                               num_workers=4)
        else:
            self.val_dataloader = DataLoader(self.val_dataset,
                                               shuffle=False,
                                               batch_size=self.batch_size,
                                               pin_memory=True,
                                               num_workers=0)

        self.decoder_learning_rate = decoder_learning_rate

        self.encoder_learning_rate = encoder_learning_rate

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=encoder_learning_rate*self.accelerator.num_processes, betas=(0.9,0.999), weight_decay=encoder_weight_decay)


        if self.load_checkpoint_bool and self.legacy_checkpoint_bool:

            self.load_checkpoint_legacy(checkpoint_path)

        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma=0.94)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 4, gamma=0.92)

        self.val_dataloader, self.train_dataloader, self.model, self.optimizer = self.accelerator.prepare(
                self.val_dataloader, self.train_dataloader, self.model, self.optimizer)

        if self.load_checkpoint_bool and not self.legacy_checkpoint_bool:

            self.accelerator.load_state(checkpoint_path)

        if self.accelerator.is_main_process:

            logging.info(self.accelerator.state)

        self.scheduler = self.accelerator.prepare(self.scheduler)


        """
        self.model = self.accelerator.unwrap_model(self.model)

        state_dict = self.model.state_dict()

        save_dict = {}
        save_dict['model']={'state_dict':state_dict}
        torch.save(save_dict, "./final_pretrain_buoy.pt")
        sys.exit(0)
        """

    def train(self):
        logging.debug("starting training")

        if self.checkpoint_epoch is None:
            start_epoch = 0
        else:
            start_epoch = self.checkpoint_epoch + 1

        for self.i_epoch in range(start_epoch, self.n_epochs):

            logging.debug(f"starting epoch {self.i_epoch}")

            self.one_epoch()

            self.scheduler.step()

            if self.i_epoch > 0 and self.i_epoch % self.save_interval == 0:

                self.accelerator.wait_for_everyone()

                self.save_model()

    def load_checkpoint_legacy(self,checkpoint_path):

        if not os.path.exists(checkpoint_path):
            logging.error(f"weight file {checkpoint_path} does not exist")
            sys.exit(0)

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        model_key = list(checkpoint.keys())[0]

        contains_decoder_hrrr = any(["decoder_hrrr" in key for key in checkpoint[model_key].keys()])

        if self.pretrain:

            reduced_keys = [key for key in checkpoint[model_key]['model_state_dict'].keys() if not "mlp" in key]

            self.model.load_state_dict({key: checkpoint[model_key]['model_state_dict'][key] for key in reduced_keys})
            
            self.i_step  = checkpoint[model_key]['step']

            self.checkpoint_epoch = checkpoint[model_key]['epoch']

            #self.optimizer.load_state_dict(checkpoint[model_key]['optimizer_state_dict'])

        elif not contains_decoder_hrrr:
            self.first_finetune_epoch = True
            """ checkpoint does not contain decoder_hrrr. It is therefore a checkpoint from pretraining.
                No epochs, steps, or optimizer values are loaded
            """

            encoder_keys = [key for key in checkpoint[model_key]['state_dict'].keys() if "encoder." in key and not "mlp" in key]
            self.model.encoder.load_state_dict({key.removeprefix('encoder.'): checkpoint[model_key]['state_dict'][key] for key in encoder_keys})

        else:
            """ checkpoint contains both encoder and decoder_hrrr, 
                suggesting that it is a checkpoint after fine-tuning has commenced
            """

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

        for self.i_batch, (sample_x, sample_y, idx) in enumerate(self.train_dataloader):

            if self.warmup and self.i_step <= self.warmup_steps:
                for g in self.optimizer.param_groups:
                    g['lr'] = (self.i_step/self.warmup_steps) * self.decoder_learning_rate

            if not self.validate:

                self.one_batch_pretrain(sample_x, sample_y, phase="train")

                if self.accelerator.is_main_process and self.i_batch % self.val_interval == 0:

                    for i,g in enumerate(self.optimizer.param_groups):
                        self.accelerator.log({f"lr-{i}":g['lr']}, step=self.i_step)

                    self.eval("val", reconstruct=False)

                    logging.info(f"VAL loss {self.global_loss.detach().mean().item()}")
                    self.accelerator.log({"val_loss":self.global_loss.detach().mean().item()}, step=self.i_step)

                if self.accelerator.is_main_process and self.i_batch % self.train_log_interval == 0:

                    logging.info(f"TRAIN loss {self.global_loss.detach().mean().item()}")
                    self.accelerator.log({"train_loss":self.global_loss.detach().mean().item()}, step=self.i_step)

            else: #self.accelerator.is_main_process:
                #else:
                self.eval("val", reconstruct=True)
                #self.accelerator.wait_for_everyone()

            #sys.exit(0)


    def eval(self, phase,reconstruct=False):

        #os.system("nvidia-smi > ./nvidia.log")

        self.model.eval()

        for batch in self.val_dataloader:

            with torch.no_grad():

                sample_x, sample_y, idx = batch

                self.one_batch_pretrain(sample_x, sample_y, phase=phase, reconstruct=reconstruct, idx=idx)

                logging.info(idx)

            if not self.validate:
                break

        self.model.train()

    def one_batch_finetune(self, sample_x, sample_y, phase="train", reconstruct=False):


        if phase == "train":

            self.reset_grad()

        """ for some reason, validate mode requires manually moving data
        over to self.device
        """

        sample_x = sample_x.to(self.device)
        sample_y = sample_y.to(self.device)

        ret = self.model(sample_x)

        loss = F.mse_loss(ret, sample_y, reduction='mean')

        """
        if phase == "val":

            sample_y = sample_y.detach().cpu().numpy()

            #np.save(f"/pscratch/sd/j/jderm/results_unet/ft4_final_{self.i_step}_{self.accelerator.process_index}.npy", y_full)

            #with open(f"/pscratch/sd/j/jderm/results_unet/ft4_final_{self.i_step}_{self.accelerator.process_index}.pkl","wb") as f:
            #    pickle.dump(list(file_names.cpu().numpy()),f)
        """


        if phase == "train":

            self.accelerator.backward(loss)

            self.optimizer.step()
        
            self.global_loss = self.accelerator.gather_for_metrics(loss)

        elif phase == "val" and reconstruct:

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

                
                if len(self.u80_list) == 24:

                    u80 = np.vstack(self.u80_list)
                    v80 = np.vstack(self.v80_list)
                    print("recon - save")
                    #np.save(f"/pscratch/sd/j/jderm/results_unet/u_results_{self.uv_iter}_{self.accelerator.process_id}.npy", u80)
                    #np.save(f"/pscratch/sd/j/jderm/results_unet/v_results_{self.uv_iter}_{self.accelerator.process_id}.npy", v80)

                    self.uv_iter += 1
                    logging.info(f"{self.uv_iter}")

                    self.u80_list = list()
                    self.v80_list = list()


            tracker=self.accelerator.get_tracker("tensorboard")
            #tracker.log_images({"u-80": new_image_ch0}, step=self.i_step)
            #tracker.log_images({"v-80": new_image_ch1}, step=self.i_step)

        elif phase=="val":
            logging.info(f"Val loss {loss.detach().item()}")
            self.accelerator.log({"val_loss":loss.detach()}, step=self.i_step)


        self.i_step += 1

    def one_batch_pretrain(self, sample_x, sample_y, phase="train", reconstruct=False, idx = None):

        sample_x = sample_x.to(self.accelerator.device)
        sample_y = sample_y.to(self.accelerator.device)

        if phase == "train":

            self.reset_grad()

        if phase == "train":

            with self.accelerator.accumulate(self.model):

                pred = self.model(sample_x)

                pred = einops.rearrange(pred[:,:,:670,:260],'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 10, p2 = 10)[:,self.patch_mask]
                sample_y = einops.rearrange(sample_y[:,:,:670,:260],'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 10, p2 = 10)[:,self.patch_mask]

                loss = F.mse_loss(pred, sample_y)

                if phase == "train":

                    self.accelerator.backward(loss)

                    self.optimizer.step()
            
                    if self.i_batch % self.train_log_interval == 0:

                        self.global_loss = self.accelerator.gather_for_metrics(loss)

        else:

            pred = self.model(sample_x)
            pred_clone = pred.clone().detach().cpu()[:,:,:670,:260]
            sample_y_clone = sample_y.clone().cpu()[:,:,:670,:260]

            pred = einops.rearrange(pred[:,:,:670,:260],'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 10, p2 = 10)[:,self.patch_mask]
            sample_y = einops.rearrange(sample_y[:,:,:670,:260],'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 10, p2 = 10)[:,self.patch_mask]
            loss = F.mse_loss(pred, sample_y)

            #y_full = y_full.detach().cpu().numpy()

            #np.save(f"/pscratch/sd/j/jderm/results_unet_july/ft_final_{self.i_step}_{self.accelerator.process_index}.npy", pred_clone.detach().cpu())
            #with open(f"/pscratch/sd/j/jderm/results_unet_july/ft_final_{self.i_step}_{self.accelerator.process_index}.pkl","wb") as f:
            #    pickle.dump(list(idx.cpu().numpy()),f)

        if phase == "val":

            logging.info(f"Val loss {loss.detach().item()}")

            self.accelerator.log({"val_loss":loss.detach().item()}, step=self.i_step)

            if reconstruct and False:

                B,C,H,W = pred_clone.shape

                new_image_ch0 = np.zeros(shape=(1,1,H,2*W))
                new_image_ch0[0,0,:H,0*W:1*W] = hrrr_mean_var[0,1]*(sample_y_clone[0,0] + hrrr_mean_var[0,0])
                new_image_ch0[0,0,:H,1*W:2*W] = hrrr_mean_var[0,1]*(pred_clone[0,0] + hrrr_mean_var[0,0])

                new_image_ch1 = np.zeros(shape=(1,1,H,2*W))
                new_image_ch1[0,0,:H,0*W:1*W] = hrrr_mean_var[1,1]*(sample_y_clone[0,1] + hrrr_mean_var[1,0])
                new_image_ch1[0,0,:H,1*W:2*W] = hrrr_mean_var[1,1]*(pred_clone[0,1] + hrrr_mean_var[1,0])

                new_image = np.sqrt(np.power(new_image_ch0,2) + np.power(new_image_ch1,2))
                new_image = new_image * 10
                new_image = np.clip(new_image,0,255)

                new_image = new_image.astype(np.uint8)

                tracker=self.accelerator.get_tracker("tensorboard")
                tracker.log_images({"u-200": new_image}, step=self.i_step)


        self.i_step += 1

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
    parser.add_argument('--validate',action="store_true")
    parser.add_argument('--npy',action="store_true")
    parser.add_argument('--dataparallel',action="store_true")
    parser.add_argument('--seed',type=int, required=False)

    parser.add_argument('--encoder-learning-rate',type=float,required=False)
    parser.add_argument('--decoder-learning-rate',type=float,required=False)
    parser.add_argument('--encoder-weight-decay',type=float,required=False)
    parser.add_argument('--decoder-weight_decay',type=float,required=False)

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

