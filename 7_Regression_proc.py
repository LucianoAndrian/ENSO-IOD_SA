"""
Regresion parcial observada
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/regre/'

# ---------------------------------------------------------------------------- #
import xarray as xr
from funciones.indices_utils import  DMI, Nino34CPC
from funciones.regre_utils import ComputeWithEffect, ComputeWithoutEffect
from funciones.general_utils import init_logger
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------- #
logger = init_logger('7_Regression_proc.py')

# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

# ---------------------------------------------------------------------------- #
dmi_or = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
n34_or = Nino34CPC(xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc"),
    start=1920, end=2020)[0]

logger.info('N34, DMI ')
dmi = dmi_or.sel(time=slice('1940-01-01', '2020-12-01'))
n34 = n34_or.sel(time=slice('1940-01-01', '2020-12-01'))

logger.info('Prec y Tref')
# pp
prec = xr.open_dataset(f'{data_dir_t_pp}ppgpcc_w_c_d_1_SON.nc')
if len(prec.sel(lat=slice(20, -60)).lat) > 0:
    prec = prec.sel(lat=slice(20, -60), lon=slice(275, 330))
else:
    prec = prec.sel(lat=slice(-60, 20), lon=slice(275, 330))

# time
time_original = prec.time

# temp
temp = xr.open_dataset(f'{data_dir_t_pp}tcru_w_c_d_0.25_SON.nc')
if len(temp.sel(lat=slice(20, -60)).lat) > 0:
    temp = temp.sel(lat=slice(20, -60), lon=slice(275, 330))
else:
    temp = temp.sel(lat=slice(-60, 20), lon=slice(275, 330))

logger.info('Regresion')
prec_n34, prec_corr_n34, prec_dmi, prec_corr_dmi, temp_n34, temp_corr_n34, \
    temp_dmi, temp_corr_dmi = \
    ComputeWithEffect(data=prec, data2=temp,
                      n34=n34.sel(time=n34.time.dt.month.isin(10)),
                      dmi=dmi.sel(time=dmi.time.dt.month.isin(10)),
                      two_variables=True,  full_season=False,
                      time_original=time_original)

prec_n34_wodmi, prec_corr_n34_wodmi, prec_dmi_won34, prec_corr_dmi_won34 = \
    ComputeWithoutEffect(data=prec,
                         n34=n34.sel(time=n34.time.dt.month.isin(10)),
                         dmi=dmi.sel(time=dmi.time.dt.month.isin(10)),
                         time_original=time_original)

temp_n34_wodmi, temp_corr_n34_wodmi, temp_dmi_won34, temp_corr_dmi_won34 = \
    ComputeWithoutEffect(data=temp,
                         n34=n34.sel(time=n34.time.dt.month.isin(10)),
                         dmi=dmi.sel(time=dmi.time.dt.month.isin(10)),
                         time_original=time_original)
logger.info('Regresion')

if save is True:
    logger.info('Saving...')
    # prec
    prec_n34.to_netcdf(f'{out_dir}prec_regre_n34.nc')
    prec_corr_n34.to_netcdf(f'{out_dir}prec_regre_n34_corr.nc')
    prec_dmi.to_netcdf(f'{out_dir}prec_regre_dmi.nc')
    prec_corr_dmi.to_netcdf(f'{out_dir}prec_regre_dmi_corr.nc')
    prec_n34_wodmi.to_netcdf(f'{out_dir}prec_regre_n34_wodmi.nc')
    prec_corr_n34_wodmi.to_netcdf(f'{out_dir}prec_regre_n34_wodmi_corr.nc')
    prec_dmi_won34.to_netcdf(f'{out_dir}prec_regre_dmi_won34.nc')
    prec_corr_dmi_won34.to_netcdf(f'{out_dir}prec_regre_dmi_won34_corr.nc')

    # temp
    temp_n34.to_netcdf(f'{out_dir}temp_regre_n34.nc')
    temp_corr_n34.to_netcdf(f'{out_dir}temp_regre_n34_corr.nc')
    temp_dmi.to_netcdf(f'{out_dir}temp_regre_dmi.nc')
    temp_corr_dmi.to_netcdf(f'{out_dir}temp_regre_dmi_corr.nc')
    temp_n34_wodmi.to_netcdf(f'{out_dir}temp_regre_n34_wodmi.nc')
    temp_corr_n34_wodmi.to_netcdf(f'{out_dir}temp_regre_n34_wodmi_corr.nc')
    temp_dmi_won34.to_netcdf(f'{out_dir}temp_regre_dmi_won34.nc')
    temp_corr_dmi_won34.to_netcdf(f'{out_dir}temp_regre_dmi_won34_corr.nc')

logger.info('HGT 200 y 750')
# HGT 200
variables = ['HGT200_SON_mer_d_w', 'HGT750_SON_mer_d_w']

hgt200 = xr.open_dataset(f'{data_dir}HGT200_SON_mer_d_w.nc')
if len(hgt200.sel(lat=slice(20, -60)).lat) > 0:
    hgt200 = hgt200.sel(lat=slice(20, -60), lon=slice(275, 330))
else:
    hgt200 = hgt200.sel(lat=slice(-60, 20), lon=slice(275, 330))
hgt200 = hgt200.sel(time=slice('1940-01-01', '2020-12-01'))
hgt200 = hgt200 - hgt200.mean('time')

# time
time_original = hgt200.time

# HGT 750
hgt750 = xr.open_dataset(f'{data_dir}HGT750_SON_mer_d_w.nc')
if len(hgt750.sel(lat=slice(20, -60)).lat) > 0:
    hgt750 = hgt750.sel(lat=slice(20, -60), lon=slice(275, 330))
else:
    hgt750 = hgt750.sel(lat=slice(-60, 20), lon=slice(275, 330))
hgt750 = hgt750.sel(time=slice('1940-01-01', '2020-12-01'))
hgt750 = hgt750 - hgt750.mean('time')

logger.info('Regresion')
hgt200_n34, hgt200_corr_n34, hgt200_dmi, hgt200_corr_dmi, hgt750_n34, \
    hgt750_corr_n34, hgt750_dmi, hgt750_corr_dmi = \
    ComputeWithEffect(data=hgt200, data2=hgt750,
                      n34=n34.sel(time=n34.time.dt.month.isin(10)),
                      dmi=dmi.sel(time=dmi.time.dt.month.isin(10)),
                      two_variables=True,  full_season=False,
                      time_original=time_original)

hgt200_n34_wodmi, hgt200_corr_n34_wodmi, hgt200_dmi_won34, \
    hgt200_corr_dmi_won34 = \
    ComputeWithoutEffect(data=hgt200,
                         n34=n34.sel(time=n34.time.dt.month.isin(10)),
                         dmi=dmi.sel(time=dmi.time.dt.month.isin(10)),
                        time_original=time_original)

hgt750_n34_wodmi, hgt750_corr_n34_wodmi, hgt750_dmi_won34, \
    hgt750_corr_dmi_won34 = \
    ComputeWithoutEffect(data=hgt750,
                         n34=n34.sel(time=n34.time.dt.month.isin(10)),
                         dmi=dmi.sel(time=dmi.time.dt.month.isin(10)),
                         time_original=time_original)
logger.info('Regresion DONE')

if save is True:
    logger.info('Saving...')
    # hgt200
    hgt200_n34.to_netcdf(f'{out_dir}hgt200_regre_n34.nc')
    hgt200_corr_n34.to_netcdf(f'{out_dir}hgt200_regre_n34_corr.nc')
    hgt200_dmi.to_netcdf(f'{out_dir}hgt200_regre_dmi.nc')
    hgt200_corr_dmi.to_netcdf(f'{out_dir}hgt200_regre_dmi_corr.nc')
    hgt200_n34_wodmi.to_netcdf(f'{out_dir}hgt200_regre_n34_wodmi.nc')
    hgt200_corr_n34_wodmi.to_netcdf(f'{out_dir}hgt200_regre_n34_wodmi_corr.nc')
    hgt200_dmi_won34.to_netcdf(f'{out_dir}hgt200_regre_dmi_won34.nc')
    hgt200_corr_dmi_won34.to_netcdf(f'{out_dir}hgt200_regre_dmi_won34_corr.nc')

    # hgt750
    hgt750_n34.to_netcdf(f'{out_dir}hgt750_regre_n34.nc')
    hgt750_corr_n34.to_netcdf(f'{out_dir}hgt750_regre_n34_corr.nc')
    hgt750_dmi.to_netcdf(f'{out_dir}hgt750_regre_dmi.nc')
    hgt750_corr_dmi.to_netcdf(f'{out_dir}hgt750_regre_dmi_corr.nc')
    hgt750_n34_wodmi.to_netcdf(f'{out_dir}hgt750_regre_n34_wodmi.nc')
    hgt750_corr_n34_wodmi.to_netcdf(f'{out_dir}hgt750_regre_n34_wodmi_corr.nc')
    hgt750_dmi_won34.to_netcdf(f'{out_dir}hgt750_regre_dmi_won34.nc')
    hgt750_corr_dmi_won34.to_netcdf(f'{out_dir}hgt750_regre_dmi_won34_corr.nc')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #