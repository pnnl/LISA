import numpy as np
from data.dataloader_nc import dataloader_superres
import matplotlib.pyplot as plt
import tqdm

import netCDF4
import xarray as xr
import os
from torch.utils.data import DataLoader

train_dataset = dataloader_superres( \
            "/pscratch/sd/j/jderm/ERA5_reduced/",
            "/pscratch/sd/j/jderm/hrrr_reduced/",
            "/global/common/software/m4506/s2s_mae/training/s2s_tar/era_alter_val.txt",
            pretraining=True)

train_dataset_multi = DataLoader(train_dataset, batch_size=48, pin_memory=True, num_workers=12)

mean = [list() for z in range(0,16)]
min = [list() for z in range(0,16)]
max = [list() for z in range(0,16)]
std = [list() for z in range(0,16)]

file_handle = open("/pscratch/sd/j/jderm/npy_val_list.csv", "a")

newpath = "/pscratch/sd/j/jderm/ERA5_NPY/"

if not os.path.exists(newpath):
    os.makedirs(newpath)

for i, (batch, _, _, fstr_list)  in enumerate(tqdm.tqdm(train_dataset_multi)):
    if i % 20 == 0:
        print(f"{i}")
        file_handle.flush()


    for j in range(0, batch.shape[0]):


        x = batch[j]
        fstr = [y[j] for y in fstr_list]

        dirstring = fstr[0].split('/')[2]
        indexstring = fstr[3]
        filestring = fstr[0].split('/')[2] + f'_{indexstring}.npy'
        csv_line = f"{dirstring}/{filestring},\n"
        print(csv_line)

        if not os.path.exists(newpath + dirstring):
            os.makedirs(newpath + dirstring)

        file_handle.write(csv_line)

        np.save(newpath + dirstring + '/' + filestring, x.numpy())

        """ 
        for q in range(0,16):
            mean[q].append(np.mean(x[q].numpy()))
            min[q].append(np.min(x[q].numpy()))
            max[q].append(np.max(x[q].numpy()))
            std[q].append(np.std(x[q].numpy()))
        """

file_handle.close()

"""
for z in range(0,16):
    print(f"[{np.mean(mean[z])},  {np.mean(std[z])}],")
"""

    
