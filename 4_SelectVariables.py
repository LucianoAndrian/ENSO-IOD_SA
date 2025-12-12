"""
Seleccion de los campos de las variables para cada caso de eventos IOD y ENSO
A partir de los sst_* salida de 2_CFSv2_DMI_N34.py ara asegurar
correspondencia entre los eventos de los índices y los campos seleccionados.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
from multiprocessing import Process
from funciones.selectvariables_utils import SelectVariables
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
from funciones.general_utils import init_logger

import warnings
warnings.simplefilter("ignore")
# ---------------------------------------------------------------------------- #
save_nc = True

cases_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
data_dir_dmi_n34 = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
out_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
# ---------------------------------------------------------------------------- #
logger = init_logger('4_SelectVariables.log')

# ---------------------------------------------------------------------------- #
# variables = ['sst']
cases = ['dmi_puros_pos', 'dmi_puros_neg', #DMI puros
         'n34_puros_pos', 'n34_puros_neg', #N34 puros
         'sim_pos', 'sim_neg', #sim misma fase
         'neutros', #neutros
         # todos de cada caso para validación
         'dmi_pos', 'dmi_neg', 'n34_pos', 'n34_neg',
         'dmi_neg_n34_pos', 'dmi_pos_n34_neg']

seasons = ['SON']

# Funcion -------------------------------------------------------------------- #
def Aux_SelectEvents(var_file, case, cases_dir, data_dir, out_dir,
                     end_file='05', index_compute=False):

    aux_cases = xr.open_dataset(f'{cases_dir}{case}_f_SON_05.nc')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]:'index'})

    data_var = xr.open_dataset(f'{data_dir}{var_file}')
    case_events = SelectVariables(aux_cases, data_var)

    var_name = var_file.split('_')[0]
    if index_compute:
        name = f'{out_dir}{var_name}_values_{case}_SON_{end_file}.nc'
    else:
        name = f'{out_dir}{var_name}_{case}_SON_{end_file}.nc'

    case_events.to_netcdf(name)

def Run(var_file, cases, div, cases_dir, data_dir, out_dir,
        end_file='05', index_compute=False):

    logger.info(f'Run()...')
    for c in range(0, len(cases), div):
        batch = cases[c:c + div]
        processes = [Process(target=Aux_SelectEvents,
                             args=(var_file, c, cases_dir, data_dir, out_dir,
                     end_file, index_compute))
                     for c in batch]

        for p in processes:
            p.start()
        for p in processes:
            p.join()

# ---------------------------------------------------------------------------- #
div = len(cases) // 2

# N34 ------------------------------------------------------------------------ #
logger.info('N34')
var_file = 'N34_SON_Leads_r_CFSv2_new.nc'
Run(var_file, cases, div,
    cases_dir=cases_dir, data_dir=data_dir_dmi_n34,
    out_dir=out_dir, index_compute=True)

# DMI ------------------------------------------------------------------------ #
logger.info('DMI')
var_file = 'DMI_SON_Leads_r_CFSv2_new.nc'
Run(var_file, cases, div,
    cases_dir=cases_dir, data_dir=data_dir_dmi_n34,
    out_dir=out_dir, index_compute=True)

# tref ----------------------------------------------------------------------- #
logger.info('Tref')
var_file = 'tref_son_detrend.nc'
Run(var_file, cases, div,
    cases_dir=cases_dir, data_dir=data_dir, out_dir=out_dir,
    index_compute=False, end_file='detrend_05')

# prec ----------------------------------------------------------------------- #
logger.info('Prec')
var_file = 'prec_son_detrend.nc'
Run(var_file, cases, div,
    cases_dir=cases_dir, data_dir=data_dir, out_dir=out_dir,
    index_compute=False, end_file='detrend_05')

# hgt750 --------------------------------------------------------------------- #
logger.info('hgt750')
var_file = 'hgt750_son_detrend.nc'
Run(var_file, cases, div,
    cases_dir=cases_dir, data_dir=data_dir, out_dir=out_dir,
    index_compute=False, end_file='detrend_05')

# hgt ------------------------------------------------------------------------ #
logger.info('hgt')
var_file = 'hgt_son.nc'
Run(var_file, cases, div,
    cases_dir=cases_dir, data_dir=data_dir, out_dir=out_dir,
    index_compute=False)

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #