import numpy as np
from data.dataloader_nc import dataloader_superres
import matplotlib.pyplot as plt
import tqdm
from models.unet import UNet
from torch.utils.data import DataLoader
import einops
import torch

#"/global/common/software/m4506/s2s_mae/training/s2s_tar/goldstandard/gold_npy_hrrr_train.csv",
train_dataset = dataloader_superres( \
            "/pscratch/sd/j/jderm/ERA5_golden/",
            "/pscratch/sd/j/jderm/HRRR_npy/",
            "/global/common/software/m4506/s2s_mae/training/s2s_tar/goldstandard/gold_pretrain_npy_pre2016.csv",
            pretraining=True,
            use_npy=True,
            normalize=False,
            two_hours=False)

train_dataloader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=12,
                              num_workers=0)

z_mean = [list() for z in range(0,14)]
z_std = [list() for z in range(0,14)]

t_mean = [list() for t in range(0,2)]
t_std = [list() for t in range(0,2)]


dataset_len = len(train_dataset) // 12 + 1

era_array = np.zeros(shape=(dataset_len, 14, 2))
hrrr_array = np.zeros(shape=(dataset_len, 2, 2))

batch = 0
        
#HRRR_sample = einops.rearrange(HRRR_sample,'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 10, p2 = 10)

accum = list()

for batch, (z, t, _) in enumerate(train_dataloader):

    accum.append(z)

    #np.save("/global/homes/j/jderm/era_sample.npy",z.numpy())
    #np.save("/global/homes/j/jderm/hrrr_sample.npy",t.numpy())
    if batch > 1000:
        break

    #z=z[:,:,:670,:260]
    #t=t[:,:,:670,:260]
    z = einops.rearrange(z, 'b c h w ->  c (b h w)')
    #t = einops.rearrange(t, 'b c hw ->  c (b hw)')
    #t = einops.rearrange(t, 'b (hw) (p1 p2 c) ->  c (b hw p1 p2)',p1=10,p2=10)

    z_mean_ten = torch.mean(z, dim=1)
    z_std_ten  = torch.std(z, dim=1)

    #t_mean_ten = torch.mean(t,dim=1)
    #t_std_ten  = torch.std(t,dim=1)

    era_array[batch,:,0] = z_mean_ten
    era_array[batch,:,1] = z_std_ten

    #hrrr_array[batch,:,0] = t_mean_ten
    #hrrr_array[batch,:,1] = t_std_ten
    if batch % 10 == 0:
        print(batch)
    if batch > 1000:
        break

#all_ = torch.vstack(accum)
#all_ = torch.median(all_,dim=0)[0]
#all_ = all_.unsqueeze(0)
#print(f"{all_.shape=}")
#torch.save(all_, "/pscratch/sd/j/jderm/median_input.pt")

era_norm = np.array(era_array[:batch+1]).mean(axis=0)
#hrrr_norm = np.array(hrrr_array[:batch+1]).mean(axis=0)

print(era_norm)
#print(hrrr_norm)

#np.save("/pscratch/sd/j/jderm/unet_buoy_norm.npy",era_norm)
#np.save("/pscratch/sd/j/jderm/unet_buoy_norm.npy",hrrr_norm)

    
