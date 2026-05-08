from accelerate import Accelerator
import torch
import argparse
import os, sys
import os.path as path
from datetime import datetime
import logging
from data.dataloader_nc import dataloader_superres
from models.models import  ERAencoder, ERAdecoder, ERA5Upscaler
from models.mae import MAE 
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

class trainer():
    def __init__(self, device, checkpoint_path, experiment_string, pretrain, epochs, batch_size, dataparallel, era5_path, hrrr_path, train_file, val_file,
                 encoder_learning_rate, encoder_weight_decay, decoder_learning_rate, decoder_weight_decay,
                 encoder_depth, encoder_dim, encoder_channels, encoder_heads, encoder_mlp_dim, 
                 decoder_depth, decoder_dim, decoder_channels, decoder_heads, decoder_mlp_dim):

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

        self.pretrain = pretrain
        self.first_finetune_epoch = False
        self.dataparallel = dataparallel

        self.batch_size = batch_size #48*4 #96*4 #40
        self.n_epochs = epochs
        self.experiment_string = experiment_string
        self.train_log_interval = 5

        self.save_interval = 3
        self.val_interval = 10

        self.i_epoch = 0
        self.i_batch = 0
        self.i_step = 0


        self.checkpoint_epoch = None
        self.validate = False

        if isinstance(device, str):
           if device =='cpu':
               device = torch.device("cpu")
           else:
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device

        self.encoder = ERAencoder(image_size=(240,440),
	       patch_size=(10,10), 
	       num_classes=192,
           channels=encoder_channels,
	       dim=encoder_dim,
	       depth=encoder_depth,
	       heads=encoder_heads,
	       mlp_dim=encoder_mlp_dim)

        if self.pretrain:
            self.model = MAE(encoder=self.encoder, decoder_dim=192, masking_ratio=0.5)

        else:
            self.hrrr_decoder = ERAdecoder(final_image_size=(670, 260),
                patch_size=(10,10), 
                final_channels=decoder_channels,
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                mlp_dim=decoder_mlp_dim) 

            self.model = ERA5Upscaler(self.encoder, self.hrrr_decoder)

        self.model = self.model.to(self.device)

        self.train_dataset = dataloader_superres( \
                    era5_path,
                    hrrr_path,
                    train_file,
                    pretraining = self.pretrain)

        self.val_dataset = dataloader_superres( \
                    era5_path,
                    hrrr_path,
                    val_file,
                    pretraining = self.pretrain)

        self.train_dataloader = DataLoader(self.train_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=8)

        self.val_dataloader = DataLoader(self.val_dataset,
                                           shuffle=True,
                                           batch_size=self.batch_size,
                                           num_workers=8)

        self.eval_datasets = {'val':cycle(self.val_dataloader)}

        if self.pretrain:

            self.opt_encoder = torch.optim.Adam(self.model.parameters(), lr=encoder_learning_rate, betas=(0.9,0.999), weight_decay=encoder_weight_decay)

        else:

            self.opt_decoder = torch.optim.Adam(self.hrrr_decoder.parameters(), lr=decoder_learning_rate , betas=(0.9,0.999), weight_decay=decoder_weight_decay)

        if not checkpoint_path == "":

            self.load_checkpoint(checkpoint_path)

        if self.dataparallel:

            self.model = torch.nn.DataParallel(self.model, device_ids = [0,1,2,3])

    def train(self):
        logging.debug("starting training")

        if self.checkpoint_epoch is None:
            start_epoch = 0
        else:
            start_epoch = self.checkpoint_epoch + 1

        for self.i_epoch in range(start_epoch, self.n_epochs):

            logging.debug(f"starting epoch {self.i_epoch}")

            self.one_epoch()

            if self.i_epoch > 0 and self.i_epoch % self.save_interval == 0:

                self.save_model()

    def save_model(self):

        current_time = datetime.now()

        checkpoint_path = f'./runs/{self.experiment_string}/'

        if not path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        model_string = f'{self.experiment_string}_'
        model_string+= f'{current_time.strftime("%Y%m%d-%H%M")}_'
        model_string+= f'epoch{self.i_epoch}.pt'

        all_models = {}
        if self.dataparallel:
            if self.pretrain:
                all_models['model'] = {
                    'epoch':self.i_epoch,
                    'step':self.i_step,
                    'model_state_dict':self.model.module.state_dict(),
                    'optimizer_state_dict_encoder':self.opt_encoder.state_dict()}
            else:
                all_models['model'] = {
                    'epoch':self.i_epoch,
                    'step':self.i_step,
                    'model_state_dict':self.model.module.state_dict(),
                    'optimizer_state_dict_decoder':self.opt_decoder.state_dict()}
        else:
            if self.pretrain:
                all_models['model'] = {
                    'epoch':self.i_epoch,
                    'step':self.i_step,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict_encoder':self.opt_encoder.state_dict()}
            else:
                all_models['model'] = {
                    'epoch':self.i_epoch,
                    'step':self.i_step,
                    'model_state_dict':self.model.state_dict(),
                    'optimizer_state_dict_decoder':self.opt_decoder.state_dict()}

        torch.save(all_models, checkpoint_path + model_string)
        logging.debug(f"model saved: {checkpoint_path + model_string}")


    def load_checkpoint(self,checkpoint_path):

        if not os.path.exists(checkpoint_path):
            logging.error(f"weight file {checkpoint_path} does not exist")
            sys.exit(0)

        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        model_key = list(checkpoint.keys())[0]

        contains_decoder_hrrr = any(["decoder_hrrr" in key for key in checkpoint[model_key].keys()])

        if self.pretrain:

            self.model.load_state_dict(checkpoint[model_key]['model_state_dict'])
            
            self.i_step  = checkpoint[model_key]['step']

            self.checkpoint_epoch = checkpoint[model_key]['epoch']

            self.opt_encoder.load_state_dict(checkpoint[model_key]['optimizer_state_dict'])


        elif not contains_decoder_hrrr:
            self.first_finetune_epoch = True
            """ checkpoint does not contain decoder_hrrr. It is therefore a checkpoint from pretraining.
                No epochs, steps, or optimizer values are loaded
            """

            encoder_keys = [key for key in checkpoint[model_key]['model_state_dict'].keys() if "encoder." in key]
            
            self.model.encoder.load_state_dict({key.removeprefix('encoder.'): checkpoint[model_key]['model_state_dict'][key] for key in encoder_keys})

            """ do not load encoder """
            #self.opt_encoder.load_state_dict(checkpoint[model_key]['optimizer_state_dict_encoder'])


        else:
            """ checkpoint contains both encoder and decoder_hrrr, 
                suggesting that it is a checkpoint after fine-tuning has commenced
            """

            decoder_keys = [key for key in checkpoint[model_key]['model_state_dict'].keys() if "decoder_hrrr." in key]
            
            self.model.decoder_hrrr.load_state_dict({key.removeprefix('decoder_hrrr.'): checkpoint[model_key]['model_state_dict'][key] for key in decoder_keys})

            self.i_step  = checkpoint[model_key]['step']

            self.checkpoint_epoch = checkpoint[model_key]['epoch']

            self.opt_decoder.load_state_dict(checkpoint[model_key]['optimizer_state_dict_decoder'])

        del checkpoint

        torch.cuda.empty_cache()

        logging.debug("model load successful")

    def one_epoch(self):

        self.eval('val')

        for self.i_batch, (sample_x, sample_y) in enumerate(self.train_dataloader):


            if not self.validate:

                if self.pretrain:
                    self.one_batch_pretrain(sample_x, sample_y, phase="train")
                else:
                    self.one_batch_finetune(sample_x, sample_y, phase="train")

                if self.i_batch % self.val_interval == 0:

                    self.eval('val')

            else:
                self.eval('val')

    def eval(self, phase):

        os.system("nvidia-smi > ./nvidia.log")
        os.system("free -h > ./mem.log")

        self.model.eval()

        dataset = self.eval_datasets[phase]

        with torch.no_grad():

            sample_x, sample_y = next(dataset)

            if self.pretrain:
                self.one_batch_pretrain(sample_x, sample_y, phase=phase)
            else:
                self.one_batch_finetune(sample_x, sample_y, phase=phase)

        self.model.train()

    def one_batch_finetune(self, sample_x, sample_y, phase="train"):

        if phase == "train":

            self.reset_grad()

        sample_x = sample_x.to(self.device)
        sample_y = sample_y.to(self.device)

        reconstruction, sample_y2 = self.model(sample_x, sample_y)

        loss = F.mse_loss(reconstruction, sample_y2)

        if phase == "train":

            loss.backward()

            self.opt_decoder.step()

            self.reset_grad()


        if phase == 'train' and self.i_step % self.train_log_interval == 0:

            logging.debug(f"TRAIN {self.i_epoch:3d} {self.i_batch:3d} | Loss {loss:.2e} ")

        if phase == 'val':

            logging.debug(f"VAL {self.i_epoch:3d} {self.i_batch:3d} | Loss {loss:.2e} ")

            gt, pred = reconstruct_image_reduced(self.train_dataset.patch_mask_indices, sample_y[0], reconstruction[0])

            gt =  gt.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()

            plot_reconstruction(gt,pred,f'./runs/{self.experiment_string}/', self.i_epoch, self.i_step)

            logging.debug("png reconstruction saved")

        self.i_step += 1

    def one_batch_pretrain(self, sample_x, sample_y, phase="train"):

        if phase == "train":

            self.reset_grad()

        sample_x = sample_x.to(self.device)

        pred_pixel_values, masked_pixels = self.model(sample_x)

        loss = F.mse_loss(pred_pixel_values, masked_pixels)

        """
        patches, masked_indices, pred_pixel_values = self.model.reconstruct(sample_x)
        img = reconstruct_image(patches, sample_x, masked_indices=masked_indices, pred_pixel_values=pred_pixel_values, patch_size=10)
        imgMasked = reconstruct_image(patches, sample_x, masked_indices=masked_indices, patch_size=10)
        sample_x=sample_x.cpu().numpy()
        img = img.detach().numpy()
        np.save("./input5.npy", sample_x)
        np.save("./input5_masked.npy", imgMasked)
        np.save("./recon5.npy", img)
        sys.exit()
        """

        if phase == "train":

            loss.backward()

            self.opt_encoder.step()

            self.reset_grad()

        if phase == 'train' and self.i_step % self.train_log_interval == 0:

            logging.debug(f"TRAIN {self.i_epoch:3d} {self.i_batch:3d} | Loss {loss:.2e} ")

        if phase == 'val':

            logging.debug(f"VAL {self.i_epoch:3d} {self.i_batch:3d} | Loss {loss:.2e} ")

        self.i_step += 1

    def reset_grad(self):

        if self.pretrain:
            self.opt_encoder.zero_grad()
        else:
            self.opt_decoder.zero_grad()


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
    parser.add_argument('--epochs',type=int,required=False)
    parser.add_argument('--pretrain',action="store_true")
    parser.add_argument('--dataparallel',action="store_true")

    parser.add_argument('--encoder-learning-rate',type=float,required=False)
    parser.add_argument('--decoder-learning-rate',type=float,required=False)
    parser.add_argument('--encoder-weight-decay',type=float,required=False)
    parser.add_argument('--decoder-weight_decay',type=float,required=False)

    parser.add_argument('--encoder-depth',type=int,required=False)
    parser.add_argument('--encoder-dim',type=int,required=False)
    parser.add_argument('--encoder-channels',type=int,required=False)
    parser.add_argument('--encoder-heads',type=int,required=False)
    parser.add_argument('--encoder-mlp_dim',type=int,required=False)

    parser.add_argument('--decoder-depth',type=int,required=False)
    parser.add_argument('--decoder-dim',type=int,required=False)
    parser.add_argument('--decoder-channels',type=int,required=False)
    parser.add_argument('--decoder-heads',type=int,required=False)
    parser.add_argument('--decoder-mlp_dim',type=int,required=False)

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

