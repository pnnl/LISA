import numpy as np
from data.dataloader_nc import dataloader_superres
import matplotlib.pyplot as plt
import tqdm


BASEDIR="/global/common/software/m4506/s2s_mae/training/s2s_tar/"
DATAFILE=BASEDIR+"goldstandard/gold_pretrain_npy_pre2016.csv"
DATAFILE=BASEDIR+"goldstandard/gold_npy_hrrr_train.csv"
VALFILE=BASEDIR+"goldstandard/gold_npy_hrrr_val.csv"

train_dataset = dataloader_superres( \
            "/pscratch/sd/j/jderm/ERA5_golden/",
            "/pscratch/sd/j/jderm/HRRR_npy/",
            DATAFILE,
            pretraining=False,
            use_npy=True,
            normalize=False,
            two_hours=True)

mean = [list() for z in range(0,2)]
min = [list() for z in range(0,2)]
max = [list() for z in range(0,2)]
std = [list() for z in range(0,2)]

for i in tqdm.tqdm(range(0,30)):
    x, _,_ = train_dataset[i]
    print(f"{x.shape=}")
    
    for q in range(0,2):
        mean[q].append(np.mean(x[q].numpy()))
        min[q].append(np.min(x[q].numpy()))
        max[q].append(np.max(x[q].numpy()))
        std[q].append(np.std(x[q].numpy()))

for z in range(0,2):
    print(f"[{np.mean(mean[z])},  {np.mean(std[z])}],")

    
