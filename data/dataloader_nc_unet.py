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

#def regridder_era5_to_hrrr(method='bilinear'):
#
#    era5_grid = xr.open_dataset('/pscratch/sd/j/jderm/ERA5_reduced/t500/200101/t.500.2001010100_2001010123.nc')['T'].isel(time=0)
#    hrrr_grid = xr.open_dataset('/global/cfs/projectdirs/m4506/yeliu/MLv1.0/mask_regridded.nc')['mask']
#
#    hrrr_grid = hrrr_grid.where(~np.isnan(hrrr_grid), 0)
#
#    regridder = xe.Regridder(era5_grid, hrrr_grid, 'bilinear')
#
#    return regridder

#old
mean_var = np.array([ \
[17.000926971435547,  14.979028701782227],
[7.649372100830078,  10.767777442932129],
[3.6418650150299072,  8.184596061706543],
[0.09457335621118546,  13.347031593322754],
[0.17622464895248413,  9.489484786987305],
[0.2089252918958664,  6.89345121383667],
[117250.8359375,  3649.899169921875],
[55379.1171875,  2158.494140625],
[29817.45703125,  1313.5845947265625],
[219.13267517089844,  4.775785446166992],
[256.5672912597656,  9.10300064086914],
[271.9204406738281,  9.631112098693848],
[98452.4921875,  5729.0849609375],
[284.3547058105469,  13.150114059448242],
[279.76373291015625,  12.817069053649902],
[2601.4658203125,  4991.43701171875]])

#new
#mean_var = np.array([ \
#[ 1.60050697e+01,  1.50534401e+01],
# [ 6.64086056e+00,  1.08392019e+01],
# [ 2.65782762e+00,  8.25013351e+00],
# [-9.50041294e-01,  1.34252491e+01],
# [-8.94404829e-01,  9.46443272e+00],
# [-8.32457960e-01,  6.90525198e+00],
# [ 1.17331102e+05,  3.70343652e+03],
# [ 5.54103125e+04,  2.17985010e+03],
# [ 2.98339258e+04,  1.32599792e+03],
# [ 2.18166061e+02,  4.67988014e+00],
# [ 2.55717987e+02,  9.17382145e+00],
# [ 2.71062469e+02,  9.75032234e+00],
# [ 9.84585078e+04,  5.73525098e+03],
# [ 2.83463074e+02,  1.32088060e+01],
# [ 2.78813660e+02,  1.28919516e+01],
# [ 2.60046558e+03,  4.99143701e+03]])

#old
hrrr_mean_var = np.array([ \
[-1.5728172063827515,  3.915628433227539],
[-1.3613358736038208,  4.049429893493652]])

#new
#hrrr_mean_var = np.array([ \
#[-1.2700492,   3.85311371],
#[-1.28516948,  3.88062514]])

#unet
mean_var = np.array([ \
 [ 1.91429816e+01, 1.46628417e+01],
 [ 9.50821559e+00, 1.00140799e+01],
 [ 4.91154847e+00, 6.62772621e+00],
 [-9.41738231e-01, 1.62353268e+01],
 [-1.05267515e+00, 1.13885355e+01],
 [-8.06062029e-01, 7.97293321e+00],
 [ 1.18107264e+05, 2.41410609e+03],
 [ 5.61505540e+04, 1.35053074e+03],
 [ 3.03372006e+04, 7.78637071e+02],
 [ 2.17359552e+02, 4.92225874e+00],
 [ 2.58621924e+02, 6.33531167e+00],
 [ 2.75126796e+02, 6.90738825e+00],
 [ 9.85641952e+04, 6.08607988e+03],
 [ 2.86475849e+02, 5.82508936e+00],
 [ 2.81413174e+02, 5.93604307e+00],
 [ 2.77966992e+03, 5.37796436e+03]])

#unet
hrrr_mean_var = np.array([ \
 [ 1.34187811,  3.89167407],
 [-2.75422505,  6.09448469]])

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
                        use_npy = False):
        """ arguments
        ERA5_root_dir (string): directory with ERA5 files
        HRRR_root_dir (string): directory with HRRR files
        correlation_file (string): path to correlation file
        """

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
                    self.observation_pairs = [(pair[0]) for pair in reader]
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


        self.era_wc_mask = np.load("/global/common/software/m4506/s2s_mae/training/s2s_tar/era_westcoast_mask.npy")
        z = einops.rearrange(self.era_wc_mask,'(h p1) (w p2) -> (h w) (p1 p2)', p1 = 10, p2 = 10)
        self.era_patch_mask = np.sum(z, axis=1) > 0
        self.era_patch_mask_indices = np.array(range(0,1056))[self.era_patch_mask]

        self.lat_indices = None
        self.lon_indices = None

        #self.regridder=regridder_era5_to_hrrr()

        self.topography_data = None
        self.topography_path = "/pscratch/sd/j/jderm/ERA5_topo/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc"
        
    def __len__(self):
        return len(self.observation_pairs)


    def __getitem__(self, idx):

        if self.pretraining:

            if not self.use_npy:
                ERA5_sample = self.read_era5(self.ERA5_root_dir, (self.observation_pairs[idx]))
                ERA5_sample = torch.as_tensor(ERA5_sample).float()
            else:
                ERA5_sample = self.read_npy(self.ERA5_root_dir, (self.observation_pairs[idx]))
                ERA5_sample = torch.as_tensor(ERA5_sample).float()

            return ERA5_sample, torch.tensor(0), torch.tensor(idx)

        else:

            """ old way"""
            if not self.use_npy:
                #ERA5_sample, HRRR_sample = self.read_era5_hrrr(self.ERA5_root_dir, self.HRRR_root_dir, self.observation_pairs[idx])
                ERA5_sample, HRRR_sample = self.read_era5(self.ERA5_root_dir, self.HRRR_root_dir, self.observation_pairs[idx])

            else:
                ERA5_sample, HRRR_sample = self.read_npy_and_hrrr(self.ERA5_root_dir, self.HRRR_root_dir, (self.observation_pairs[idx]))

            ERA5_sample = torch.as_tensor(ERA5_sample).float()
            HRRR_sample = torch.as_tensor(HRRR_sample).float()

            """ The HRRR sample is served in patches. We will not need to un-patch the HRRR target and 
                model output to perform a loss.
            """

            #HRRR_sample = einops.rearrange(HRRR_sample,'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1 = 10, p2 = 10)
            ##HRRR_sample = einops.rearrange(HRRR_sample,'c (h p1) (w p2) -> (h w) c (p1 p2)', p1 = 10, p2 = 10)
            #HRRR_sample = HRRR_sample[self.patch_mask]
            ##HRRR_sample = einops.rearrange(HRRR_sample,'(hw) c (p1 p2) -> c (hw p1 p2)', p1 = 10, p2 = 10)
            #ERA5_sample = torch.rand((16,704,704))
            #HRRR_sample = torch.rand((2,704,704))

            return ERA5_sample, HRRR_sample, torch.tensor(idx)

    def read_npy_and_hrrr(self,path,hrrr_path,observation):

        uvzt_stack = np.zeros(shape=(16,704,704),dtype=np.float32)

        uvzt_stack[:,:670,:260] = np.load(path+observation[0])

        p_index = 0

        for i in range(0,uvzt_stack.shape[0]):

            arr = uvzt_stack[i]

            arr  = arr - mean_var[p_index,0]
            #arr  = arr + 1 - mean_var[p_index,0]
            arr /= mean_var[p_index,1]

            uvzt_stack[p_index] = arr

            p_index += 1


        """ hrrr """

        hrrr_uvzt_stack = np.zeros(shape=(2,704,704))

        hrrr_index = int(observation[1])

        yyyymm_ = observation[0].split('/')[0]

        hrrr_file = f"{yyyymm_}/{yyyymm_}_{hrrr_index}.npy"


        arr_all_channels = np.load(hrrr_path+hrrr_file)

        p_index = 0

        for p_index in range(0,hrrr_uvzt_stack.shape[0]):

            #arr  = arr + 1 - hrrr_mean_var[p_index,0]
            arr  = arr_all_channels[p_index]
            arr  = arr - hrrr_mean_var[p_index,0]
            arr /= hrrr_mean_var[p_index,1]
            hrrr_uvzt_stack[p_index][:670,:260] = arr

        #Path(f"/pscratch/sd/j/jderm/HRRR_npy/{yyyymm_}").mkdir(parents=True, exist_ok=True)
        #np.save(f"/pscratch/sd/j/jderm/HRRR_npy/{yyyymm_}/{yyyymm_}_{hrrr_index}.npy",hrrr_uvzt_stack)


        return uvzt_stack, hrrr_uvzt_stack


    def read_npy(self,path,observation):

        uvzt_stack = np.load(path+observation)

        p_index = 0

        for i in range(0,uvzt_stack.shape[0]):

            arr = uvzt_stack[i]

            arr  = arr + 1 - mean_var[p_index,0]
            #arr  = arr  - mean_var[p_index,0]
            arr /= mean_var[p_index,1]

            uvzt_stack[p_index] = arr

            p_index += 1

        return uvzt_stack

    #def read_era5(self,path,observation): UNET
    def read_era5(self,path, hrrr_path , observation):

        basestring = observation[0]
        era5_hour = int(observation[1])
        hrrr_file = observation[2]
        hrrr_index = int(observation[3])
        era_alt_1 = observation[4]
        era_alt_2 = observation[5]
        era_alt_3 = observation[6]
        alt_index = int(observation[7])

        yearmo = basestring.split('/')[2]
        alt_file_name = f"/{yearmo}_{alt_index}.npy"
        era_alt_path = f"/pscratch/sd/j/jderm/ERA5_unet/{yearmo}"

        paramlist = ['u','v','z','t']
        heightlist = ['200','500','700']
        #hour = np.random.choice(range(0,23))

        lat_min, lat_max = 10, 70
        lon_min, lon_max = -150,-40

        p_index = 0

        uvzt_stack = np.zeros(shape=(16,670,260))
        
        for param in paramlist:
            for height in heightlist:
                pfile = basestring.replace("z200","".join([param,height]))
                pfile = pfile.replace("z.200","".join([param,".",height]))

                ds = xr.open_dataset(path+pfile, engine="netcdf4")

                regrid = self.regridder(ds[param.upper()][era5_hour])
                arr  = regrid.values

                """
                    We normalize each channel to be mean zero with std 1. Early tests
                    suggested that this is very important.
                """
                #arr  = arr + 1 - mean_var[p_index,0]
                #arr /= mean_var[p_index,1]

                uvzt_stack[p_index] = arr

                p_index += 1

        lat_min, lat_max = 10, 70
        lon_min, lon_max = 210,320

        self.lat_indices = None
        self.lon_indices = None

        """
        These three files need to be treated differently. For some reason, the longitude
        can only be indexed 0 to 360, instead of 180W to 180E
        """

        for param,era_file in [("sp",era_alt_1), ("var_2t",era_alt_2), ("var_2d",era_alt_3)]:

            file_with_base = "/pscratch/sd/j/jderm/ERA5_alternate/" + era_file

            ds = xr.open_dataset(file_with_base, engine="netcdf4")

            regrid = self.regridder(ds[param.upper()][alt_index])
            arr  = regrid.values


            #arr  = arr + 1 - mean_var[p_index,0]
            #arr /= mean_var[p_index,1]

            uvzt_stack[p_index] = arr
            p_index += 1


        """
        The topography CDF can be accessed using the coordinates from the
        above ERA5 cdf read.
        """

        if self.topography_data is None:
            ds = xr.open_dataset(self.topography_path, engine="netcdf4")

            regrid = self.regridder(ds['Z'][0])
            arr  = regrid.values

            #arr  = arr + 1 - mean_var[p_index,0]
            #arr /= mean_var[p_index,1]

            uvzt_stack[p_index]  = arr
            self.topography_data = arr

            p_index += 1

        else:
            uvzt_stack[p_index] = self.topography_data
            p_index += 1

        

        """ hrrr """ #UNET!
        alt_file_name = f"/{yearmo}_{alt_index}.npy"
        era_alt_path = f"/pscratch/sd/j/jderm/ERA5_unet/{yearmo}"
        np.save(era_alt_path+alt_file_name,uvzt_stack.astype(np.float32))

        with open("/global/homes/j/jderm/gold_npy.csv","a") as f:
            f.write(f"{yearmo}/{yearmo}_{alt_index}.npy,{hrrr_index},\n")

        hrrr_uvzt_stack = np.zeros(shape=(2,670,260))
        """
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


        #return uvzt_stack UNET!
        return uvzt_stack, hrrr_uvzt_stack


    def read_era5_hrrr(self, ERA5_path, HRRR_path, observation):

        era5_file = observation[0]
        era5_hour = int(observation[1])
        hrrr_file = observation[2]
        hrrr_index = int(observation[3])

        paramlist = ['u','v','z','t']
        heightlist = ['200','500','700']


        lat_min, lat_max = 10, 70
        lon_min, lon_max = -150,-40


        p_index = 0

        uvzt_stack = np.zeros(shape=(12,240,440))

        """ era5 """
        
        for param in paramlist:
            for height in heightlist:
                pfile = era5_file.replace("z200","".join([param,height]))
                pfile = pfile.replace("z.200","".join([param,".",height]))

                ds = xr.open_dataset(ERA5_path+pfile, engine="netcdf4")

                if self.lat_indices is None:
                    lat = ds.latitude
                    lon = ds.longitude

                    self.lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                    self.lon_indices = np.where((lon >= lon_min) & (lon <= lon_max))[0]

                arr  = ds[param.upper()][era5_hour][0][self.lat_indices[0]:self.lat_indices[-1],self.lon_indices[0]:self.lon_indices[-1]].values
                arr  = arr - mean_var[p_index,0]
                arr /= mean_var[p_index,1]
                uvzt_stack[p_index] = arr
                p_index += 1


        """ hrrr """

        hrrr_uvzt_stack = np.zeros(shape=(2,670,260))
        p_index = 0
        for param in ["u","v"]:
            pfile = hrrr_file.replace("v",param)

            ds = xr.open_dataset(HRRR_path+pfile, engine="netcdf4")

            arr  = ds[param][hrrr_index].values
            arr  = arr - hrrr_mean_var[p_index,0]
            arr /= hrrr_mean_var[p_index,1]
            hrrr_uvzt_stack[p_index] = arr
            p_index += 1


        return uvzt_stack, hrrr_uvzt_stack


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dl = dataloader_superres("/pscratch/sd/j/jderm/ERA5_reduced/",
                             "/pscratch/sd/j/jderm/HRRR_reduced/",
                             "../era_hrrr_train.txt",
                             False)
    
    dataloader = DataLoader(dl,batch_size=10, shuffle=True, num_workers=32)


