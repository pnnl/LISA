import torch
from torch.utils.data import Dataset, DataLoader
from calendar import monthrange
import datetime
from datetime import timezone
import numpy as np
from pathlib import Path
import netCDF4
import xarray as xr
import csv
import einops
#import xesmf as xe
import os

#was here before
mean_var = np.array([ \
[17.113845825195312,  15.029386520385742],
[7.700785160064697,  10.803210258483887],
[3.6761837005615234,  8.208903312683105],
[0.09876050055027008,  13.40981388092041],
[0.17571690678596497,  9.512951850891113],
[0.2075476497411728,  6.907904148101807],
[117233.3125,  3677.968017578125],
[55366.953125,  2172.09716796875],
[29808.826171875,  1321.1549072265625],
[219.0911865234375,  4.745963096618652],
[256.5377502441406,  9.165780067443848],
[271.87542724609375,  9.691925048828125],
[98448.7421875,  5729.951171875],
[284.2828369140625,  13.214090347290039],
[279.71063232421875,  12.88288402557373],
[2601.4658203125,  4991.43701171875]])

#buoy
mean_var = np.array([ \
[ 1.60050697e+01,  1.50534401e+01],
 [ 6.64086056e+00,  1.08392019e+01],
 [ 2.65782762e+00,  8.25013351e+00],
 [-9.50041294e-01,  1.34252491e+01],
 [-8.94404829e-01,  9.46443272e+00],
 [-8.32457960e-01,  6.90525198e+00],
 [ 1.17331102e+05,  3.70343652e+03],
 [ 5.54103125e+04,  2.17985010e+03],
 [ 2.98339258e+04,  1.32599792e+03],
 [ 2.18166061e+02,  4.67988014e+00],
 [ 2.55717987e+02,  9.17382145e+00],
 [ 2.71062469e+02,  9.75032234e+00],
 [ 9.84585078e+04,  5.73525098e+03],
 [ 2.83463074e+02,  1.32088060e+01],
 [ 2.78813660e+02,  1.28919516e+01],
 [ 2.60046558e+03,  4.99143701e+03]])
"""

#just remeasured May21
mean_var = np.array( \
[[1.70979243e+01, 1.61520148e+01],
 [7.68642593e+00, 1.13181841e+01],
 [3.66457485e+00, 8.45542909e+00],
 [8.75418008e-02, 1.37348908e+01],
 [1.40072843e-01, 9.75367459e+00],
 [1.84629543e-01, 7.07008378e+00],
 [1.17140528e+05, 4.12760500e+03],
 [5.53173359e+04, 2.41824421e+03],
 [2.97823413e+04, 1.46084399e+03],
 [2.18908956e+02, 4.99909595e+00],
 [2.56307285e+02, 1.02458518e+01],
 [2.71620696e+02, 1.08832846e+01],
 [9.83572521e+04, 5.73648014e+03],
 [2.83998821e+02, 1.49196428e+01]])

"""
hrrr_mean_var = np.array([ \
[1.5682458877563477,  3.0335888862609863],
[-3.9326555728912354,  4.704221725463867]])

#bouy?
hrrr_mean_var = np.array([ \
 [ 1.62887898,  3.81834122],
 [-4.01411624,  6.61316559]])

hrrr_mask = np.load("/global/common/software/m4506/s2s_mae/training/s2s_tar/hrrr_mask_reduced.npy")
era_wc_mask = np.load("/global/common/software/m4506/s2s_mae/training/s2s_tar/era_westcoast_mask.npy")

"""
'(h w) (p1 p2 c) -> c (h p1) (w p2)
"""
class dataloader_superres(Dataset):

    def __init__(self,  ERA5_root_dir="/qfs/projects/windpower_wfip2uq/derm950/ERA5_S2S/",
                        HRRR_root_dir="/qfs/projects/windpower_wfip2uq/derm950/HRRR/",
                        correlation_file=None,
                        pretraining=True,
                        use_npy = False,
                        normalize=True,
                        two_hours=False):
        """ arguments
        ERA5_root_dir (string): directory with ERA5 files
        HRRR_root_dir (string): directory with HRRR files
        correlation_file (string): path to correlation file
        """

        self.two_hours = two_hours
        if self.two_hours:
            self.num_timesteps = 4
        else:
            self.num_timesteps = 1

        self.ERA5_root_dir = ERA5_root_dir
        self.HRRR_root_dir = HRRR_root_dir

        self.use_npy = use_npy
        
        self.correlation_file = correlation_file
        self.pretraining = pretraining

        if self.pretraining:
            with open(self.correlation_file, newline='') as f:
                reader = csv.reader(f)
                if not self.use_npy:
                    self.observation_pairs = [(pair[0],pair[1],pair[2],pair[3],pair[4],pair[5],pair[6],pair[7]) for pair in reader]
                else:
                    self.observation_pairs = [(pair[0],pair[1]) for pair in reader]
        else:
            with open(self.correlation_file, newline='') as f:
                reader = csv.reader(f)
                if not self.use_npy:
                    self.observation_pairs = [(pair[0],pair[1],pair[2],pair[3],pair[4],pair[5],pair[6],pair[7]) for pair in reader]
                else:
                    self.observation_pairs = [(pair[0],pair[1]) for pair in reader]
                
        self.hrrr_mask = np.load("/global/common/software/m4506/s2s_mae/training/s2s_tar/hrrr_mask_reduced.npy")
        z = einops.rearrange(self.hrrr_mask,'(h p1) (w p2) -> (h w) (p1 p2)', p1 = 10, p2 = 10)
        self.patch_mask = np.sum(z, axis=1) > 0
        self.patch_mask_indices = np.array(range(0,1742))[self.patch_mask]
        self.patch_mask_anti_indices = np.array(range(0,1742))[~self.patch_mask]


        self.era_wc_mask = np.load("/global/common/software/m4506/s2s_mae/training/s2s_tar/era_westcoast_mask.npy")
        z = einops.rearrange(self.era_wc_mask,'(h p1) (w p2) -> (h w) (p1 p2)', p1 = 10, p2 = 10)
        self.era_patch_mask = np.sum(z, axis=1) > 0
        self.era_patch_mask_indices = np.array(range(0,1056))[self.era_patch_mask]

        self.lat_indices = None
        self.lon_indices = None

        self.topography_data = None
        self.topography_path = "/pscratch/sd/j/jderm/ERA5_topo/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
        self.normalize = normalize
        
    def __len__(self):
        return len(self.observation_pairs)

    def __getitem__(self, idx):

        if self.pretraining:

            if not self.use_npy:
                ERA5_sample = self.read_era5_netcdf(self.ERA5_root_dir, (self.observation_pairs[idx]))

                if self.num_timesteps > 1:
                    ERA5_samples_list = [torch.as_tensor(ERA5_sample).float()]
                    for i in range(1, self.num_timesteps):
                        ERA5_sample_prev = self.read_era5_netcdf(self.ERA5_root_dir, (self.observation_pairs[idx-i]))
                        ERA5_samples_list.append(torch.as_tensor(ERA5_sample_prev).float())
                    ERA5_sample = torch.cat(ERA5_samples_list, dim=0)
                else:
                    ERA5_sample = torch.as_tensor(ERA5_sample).float()
            else:
                ERA5_sample = self.read_era5_npy(self.ERA5_root_dir, (self.observation_pairs[idx]))

                if self.num_timesteps > 1:
                    ERA5_samples_list = [torch.as_tensor(ERA5_sample).float()[:15,...]]  # discard last channel
                    for i in range(1, self.num_timesteps):
                        ERA5_sample_prev = self.read_era5_npy(self.ERA5_root_dir, (self.observation_pairs[idx-i]))
                        ERA5_samples_list.append(torch.as_tensor(ERA5_sample_prev).float()[:15,...])
                    ERA5_sample = torch.cat(ERA5_samples_list, dim=0)
                else:
                    ERA5_sample = torch.as_tensor(ERA5_sample).float()
                    ERA5_sample = ERA5_sample[:15,...]  # discard last channel

            return ERA5_sample, torch.tensor(0), torch.tensor(idx)

        else:

            """ old way"""
            if not self.use_npy:
                ERA5_sample = self.read_era5_netcdf(self.ERA5_root_dir, (self.observation_pairs[idx]))
                HRRR_sample = self.read_hrrr_netcdf(self.HRRR_root_dir, (self.observation_pairs[idx]))
                if self.num_timesteps > 1:
                    ERA5_samples_list = [torch.as_tensor(ERA5_sample).float()]
                    for i in range(1, self.num_timesteps):
                        ERA5_sample_prev = self.read_era5_netcdf(self.ERA5_root_dir, (self.observation_pairs[idx-i]))
                        ERA5_samples_list.append(torch.as_tensor(ERA5_sample_prev).float())

            else:
                ERA5_sample = self.read_era5_npy(self.ERA5_root_dir, (self.observation_pairs[idx]))
                HRRR_sample = self.read_hrrr_npy(self.HRRR_root_dir, (self.observation_pairs[idx]))
                if self.num_timesteps > 1:
                    ERA5_samples_list = [torch.as_tensor(ERA5_sample).float()]
                    for i in range(1, self.num_timesteps):
                        ERA5_sample_prev = self.read_era5_npy(self.ERA5_root_dir, (self.observation_pairs[idx-i]))
                        ERA5_samples_list.append(torch.as_tensor(ERA5_sample_prev).float())

            if self.num_timesteps == 1:
                ERA5_sample = torch.as_tensor(ERA5_sample).float()
            HRRR_sample = torch.as_tensor(HRRR_sample).float()

            """ The HRRR sample is served in patches. We will not need to un-patch the HRRR target and
                model output to perform a loss.
            """

            if self.num_timesteps == 1:
                ERA5_sample = ERA5_sample[:15,...]  # discard last channel
            else:
                # Apply channel filtering and concatenate
                ERA5_samples_filtered = []
                for era5_tensor in ERA5_samples_list:
                    ERA5_samples_filtered.append(era5_tensor[:15,...])  # discard last channel
                ERA5_sample = torch.cat(ERA5_samples_filtered, dim=0)

            HRRR_sample = einops.rearrange(HRRR_sample,'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 10, p2 = 10)
            HRRR_sample = HRRR_sample[self.patch_mask]

            return ERA5_sample, HRRR_sample, torch.tensor(idx)


    def read_npy_and_hrrr(self,path,hrrr_path,observation):


        uvzt_stack = np.load(path+observation[0])

        if self.normalize:

            for p_index in range(0,uvzt_stack.shape[0]):

                arr = uvzt_stack[p_index]

                #arr  = arr - mean_var[p_index,0]
                arr  = arr + 1 - mean_var[p_index,0]
                arr /= mean_var[p_index,1]

                uvzt_stack[p_index] = arr

        """ hrrr """

        hrrr_index = int(observation[1])

        yyyymm_ = observation[0].split('/')[0]

        hrrr_file = f"{yyyymm_}/{yyyymm_}_{hrrr_index}.npy"

        hrrr_uvzt_stack = np.load(hrrr_path+hrrr_file)

        if self.normalize:

            for p_index in range(0,hrrr_uvzt_stack.shape[0]):

                arr  = hrrr_uvzt_stack[p_index]
                arr  = arr - hrrr_mean_var[p_index,0]
                arr /= hrrr_mean_var[p_index,1]
                hrrr_uvzt_stack[p_index] = arr
                p_index += 1

        #Path(f"/pscratch/sd/j/jderm/HRRR_npy/{yyyymm_}").mkdir(parents=True, exist_ok=True)
        #np.save(f"/pscratch/sd/j/jderm/HRRR_npy/{yyyymm_}/{yyyymm_}_{hrrr_index}.npy",hrrr_uvzt_stack)

        return uvzt_stack, hrrr_uvzt_stack


    def read_npy(self,path,observation):

        uvzt_stack = np.load(path+observation)


        for p_index in range(0,uvzt_stack.shape[0]):

            arr = uvzt_stack[i]

            arr  = arr + 1 - mean_var[p_index,0]
            #arr  = arr  - mean_var[p_index,0]
            arr /= mean_var[p_index,1]

            uvzt_stack[p_index] = arr

        return uvzt_stack

    def read_era5(self,path,observation):

        basestring = observation[0]
        era5_hour = int(observation[1])
        hrrr_file = observation[2]
        hrrr_index = "NA" #int(observation[3])
        era_alt_1 = observation[4]
        era_alt_2 = observation[5]
        era_alt_3 = observation[6]
        alt_index = int(observation[7])

        yearmo = basestring.split('/')[2]
        alt_file_name = f"/{yearmo}_{alt_index}.npy"
        era_alt_path = f"/pscratch/sd/j/jderm/ERA5_golden/{yearmo}"

        paramlist = ['u','v','z','t']
        heightlist = ['200','500','700']
        #hour = np.random.choice(range(0,23))

        lat_min, lat_max = 10, 70
        lon_min, lon_max = -150,-40

        p_index = 0

        uvzt_stack = np.zeros(shape=(16,240,440))

        if Path(era_alt_path + alt_file_name).is_file():

            return uvzt_stack, np.zeros(shape=(2,10,10))


        if True:
            
            for param in paramlist:
                for height in heightlist:
                    pfile = basestring.replace("z200","".join([param,height]))
                    pfile = pfile.replace("z.200","".join([param,".",height]))

                    ds = xr.open_dataset(path+pfile, engine="netcdf4")

                    lat = ds.latitude
                    lon = ds.longitude

                    lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                    lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                    arr  = ds[param.upper()][era5_hour][0][lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]].values

                    #regrid = self.regridder(ds[param.upper()][era5_hour])

                    #arr  = ds[param.upper()][era5_hour]

                    """
                        We normalize each channel to be mean zero with std 1. Early tests
                        suggested that this is very important.
                    """

                    if self.normalize:
                        arr  = arr + 1 - mean_var[p_index,0]
                        arr /= mean_var[p_index,1]

                    uvzt_stack[p_index] = arr

                    p_index += 1

            lat_min, lat_max = 10, 70
            lon_min, lon_max = 210,320

            """
            These three files need to be treated differently. For some reason, the longitude
            can only be indexed 0 to 360, instead of 180W to 180E
            """

            for param,era_file in [("sp",era_alt_1), ("var_2t",era_alt_2), ("var_2d",era_alt_3)]:

                file_with_base = "/pscratch/sd/j/jderm/ERA5_alternate/" + era_file

                ds = xr.open_dataset(file_with_base, engine="netcdf4")
                lat = ds.latitude
                lon = ds.longitude

                lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                #regrid = self.regridder(ds[param.upper()][alt_index])

                arr  = ds[param.upper()][alt_index][lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]].values

                if self.normalize:
                    arr  = arr + 1 - mean_var[p_index,0]
                    arr /= mean_var[p_index,1]

                uvzt_stack[p_index] = arr
                p_index += 1


            """
            The topography CDF can be accessed using the coordinates from the
            above ERA5 cdf read.
            """

            if self.topography_data is None:
                ds = xr.open_dataset(self.topography_path, engine="netcdf4")
                lat = ds.latitude
                lon = ds.longitude

                lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]


                arr = ds['Z'][0][lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]].values

                #regrid = self.regridder(ds['Z'][0])

                if self.normalize:
                    arr  = arr + 1 - mean_var[p_index,0]
                    arr /= mean_var[p_index,1]

                uvzt_stack[p_index]  = arr

                self.topography_data = arr

                p_index += 1

            else:
                uvzt_stack[p_index] = self.topography_data

                p_index += 1

        
        with open("/global/homes/j/jderm/gold_pretrain_npy2.csv","a") as f:
            f.write(f"{yearmo}/{yearmo}_{alt_index}.npy,{hrrr_index},\n")
        Path(f"{era_alt_path}").mkdir(parents=True, exist_ok=True)
        np.save(era_alt_path+alt_file_name,uvzt_stack.astype(np.float32))

        hrrr_uvzt_stack = np.zeros(shape=(2,10,10))
        """ hrrr """ #UNET!
        """
        alt_file_name = f"/{yearmo}_{alt_index}.npy"
        era_alt_path = f"/pscratch/sd/j/jderm/ERA5_golden/{yearmo}"
        #np.save(era_alt_path+alt_file_name,uvzt_stack.astype(np.float32))

        with open("/global/homes/j/jderm/gold_pretrain_npy.csv","a") as f:
            f.write(f"{yearmo}/{yearmo}_{alt_index}.npy,{hrrr_index},\n")

        hrrr_uvzt_stack = np.zeros(shape=(2,670,260))

        p_index = 0
        for param in ["u","v"]:
            pfile = hrrr_file.replace("u",param)

            ds = xr.open_dataset(hrrr_path+pfile, engine="netcdf4")

            arr  = ds[param][hrrr_index].values

            #arr  = arr - hrrr_mean_var[p_index,0]
            #arr /= hrrr_mean_var[p_index,1]

            hrrr_uvzt_stack[p_index] = arr
            p_index += 1
        """

        return uvzt_stack, hrrr_uvzt_stack

    def read_hrrr_npy(self, hrrr_path, observation):
        """ hrrr """

        hrrr_index = int(observation[1])

        yyyymm_ = observation[0].split('/')[0]

        hrrr_file = f"{yyyymm_}/{yyyymm_}_{hrrr_index}.npy"

        hrrr_uvzt_stack = np.load(hrrr_path+hrrr_file)

        if self.normalize:

            for p_index in range(0,hrrr_uvzt_stack.shape[0]):

                arr  = hrrr_uvzt_stack[p_index]

                arr  = arr - hrrr_mean_var[p_index,0]
                arr /= hrrr_mean_var[p_index,1]

                hrrr_uvzt_stack[p_index] = arr

        return hrrr_uvzt_stack


    def read_era5_npy(self,era5_path,observation):

        uvzt_stack = np.load(era5_path+observation[0])[:15]

        if self.normalize:

            for p_index in range(0,uvzt_stack.shape[0]):

                arr = uvzt_stack[p_index]

                arr  = arr - mean_var[p_index,0]
                arr /= mean_var[p_index,1]

                uvzt_stack[p_index] = arr


        return uvzt_stack

    def read_era5_netcdf(self,path,observation):

        basestring = observation[0]
        era5_hour = int(observation[1])
        #hrrr_file = observation[2]
        #hrrr_index = int(observation[3])
        era_alt_1 = observation[4]
        era_alt_2 = observation[5]
        era_alt_3 = observation[6]
        alt_index = int(observation[7])

        paramlist = ['u','v','z','t']
        heightlist = ['200','500','700']

        lat_min, lat_max = 10, 70
        lon_min, lon_max = -150,-40

        p_index = 0

        uvzt_stack = np.zeros(shape=(16,240,440))
        
        for param in paramlist:
            for height in heightlist:
                pfile = basestring.replace("z200","".join([param,height]))
                pfile = pfile.replace("z.200","".join([param,".",height]))

                ds = xr.open_dataset(path+pfile, engine="netcdf4")

                lat = ds.latitude
                lon = ds.longitude

                lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                arr  = ds[param.upper()][era5_hour][0][lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]].values

                if self.normalize:
                    arr  = arr + 1 - mean_var[p_index,0]
                    arr /= mean_var[p_index,1]

                uvzt_stack[p_index] = arr

                p_index += 1

        lat_min, lat_max = 10, 70
        lon_min, lon_max = 210,320

        for param,era_file in [("sp",era_alt_1), ("var_2t",era_alt_2), ("var_2d",era_alt_3)]:

            file_with_base = "/pscratch/sd/j/jderm/ERA5_alternate/" + era_file

            ds = xr.open_dataset(file_with_base, engine="netcdf4")
            lat = ds.latitude
            lon = ds.longitude

            lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
            lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

            arr  = ds[param.upper()][alt_index][lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]].values

            if self.normalize:
                arr  = arr + 1 - mean_var[p_index,0]
                arr /= mean_var[p_index,1]

            uvzt_stack[p_index] = arr
            p_index += 1


        """
        The topography CDF can be accessed using the coordinates from the
        above ERA5 cdf read.
        """

        if self.topography_data is None:
            ds = xr.open_dataset(self.topography_path, engine="netcdf4")
            lat = ds.latitude
            lon = ds.longitude

            lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
            lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

            arr = ds['Z'][0][lat_indices[0]:lat_indices[-1],lon_indices[0]:lon_indices[-1]].values

            if self.normalize:
                arr  = arr + 1 - mean_var[p_index,0]
                arr /= mean_var[p_index,1]

            uvzt_stack[p_index]  = arr

            self.topography_data = arr

            p_index += 1

        else:

            uvzt_stack[p_index] = self.topography_data

            p_index += 1

    
    def read_hrrr_netcdf(self,hrrr_path,observation):

        basestring = observation[0]
        era5_hour = int(observation[1])
        hrrr_file = observation[2]
        hrrr_index = int(observation[3])
        era_alt_1 = observation[4]
        era_alt_2 = observation[5]
        era_alt_3 = observation[6]
        alt_index = int(observation[7])

        hrrr_uvzt_stack = np.zeros(shape=(2,670,260))

        p_index = 0

        for param in ["u","v"]:

            pfile = hrrr_file.replace("u",param)

            ds = xr.open_dataset(hrrr_path+pfile, engine="netcdf4")

            arr  = ds[param][hrrr_index].values

            if self.normalize:

                arr  = arr - hrrr_mean_var[p_index,0]

                arr /= hrrr_mean_var[p_index,1]

            hrrr_uvzt_stack[p_index] = arr

            p_index += 1

        return hrrr_uvzt_stack



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dl = dataloader_superres("/pscratch/sd/j/jderm/ERA5_reduced/",
                             "/pscratch/sd/j/jderm/HRRR_reduced/",
                             "../era_hrrr_train.txt",
                             False)
    
    dataloader = DataLoader(dl,batch_size=10, shuffle=True, num_workers=32)


