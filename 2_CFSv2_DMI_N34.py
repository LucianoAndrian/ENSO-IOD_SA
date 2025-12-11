"""
Calculo de Niño3.4 y DMI a partir del preprocesamiento en 1_CFSv2_preSelect.py
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
from funciones.general_utils import init_logger

import warnings
warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------- #
case_field_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
out_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
save_nc = True

# ---------------------------------------------------------------------------- #
logger = init_logger('2_CFSv2_DMI_N34.log')

# ---------------------------------------------------------------------------- #
logger.info('Open SST data from 1_CFSv2_preSelect.py')
sst = xr.open_dataset(f'{case_field_dir}sst_son_detrend.nc')

# ---------------------------------------------------------------------------- #
logger.info('Compute indices')

# Niño 3.4/ONI
n34_son = sst.sel(lat=slice(-4, 4),
                        lon=slice(190, 240)).mean(['lon', 'lat'])

# IODW, IODE --> DMI
iodw = sst.sel(lon=slice(50, 70), lat=slice(-10, 10)).mean(['lon', 'lat'])
iode = sst.sel(lon=slice(90, 110), lat=slice(-10, 0)).mean(['lon', 'lat'])
dmi_son = iodw - iode

# ---------------------------------------------------------------------------- #
if save_nc is True:
    logger.info('savings...')
    n34_son.to_netcdf(f'{out_dir}N34_SON_Leads_r_CFSv2_new.nc')
    dmi_son.to_netcdf(f'{out_dir}DMI_SON_Leads_r_CFSv2_new.nc')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #