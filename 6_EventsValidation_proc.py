"""
Validación ENSO IOD CFSv2
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/'

variables = ['prec', 'tref']
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from scipy.stats import ttest_ind
from funciones.general_utils import init_logger, Weights

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------- #
logger = init_logger('6_EventsValidation_proc.log')

# ---------------------------------------------------------------------------- #
nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/' #fechas
# obs
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados

# CFSv2 # fechas ya seleccionadas
cases_dir = '/pikachu/datos/luciano.andrian/cases_fields/'

# Funciones ------------------------------------------------------------------ #
def CompositeSimple_obs(original_data, index, mmin, mmax, mean=True):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(
            time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(month=comp_field['time.month'], mmin=mmin, mmax=mmax))

        if len(comp_field.time) != 0:
            if mean:
                comp_field = comp_field.mean(['time'], skipna=True)
            else:
                pass
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field
    else:
        print(' len index = 0')

def CaseComp_obs(data, s, mmonth, c, two_variables=False, data2=None):
    """
    Las fechas se toman del periodo 1920-2020 basados en el DMI y N34 con ERSSTv5
    Cuando se toman los periodos 1920-1949 y 1950_2020 las fechas que no pertencen
    se excluyen de los composites en CompositeSimple()
    """
    mmin = mmonth[0]
    mmax = mmonth[-1]

    aux = xr.open_dataset(nc_date_dir + '1920_2020' + '_' + s + '.nc')
    neutro = aux.Neutral

    try:
        case = aux[c]
        case = case.where(case >= 1982)
        neutro = neutro.where(neutro>=1982)
        aux.close()

        case_num = len(case.values[np.where(~np.isnan(case.values))])

        neutro_comp = CompositeSimple_obs(original_data=data, index=neutro,
                                      mmin=mmin, mmax=mmax, mean=True)
        data_comp = CompositeSimple_obs(original_data=data, index=case,
                                    mmin=mmin, mmax=mmax, mean=False)

        comp = data_comp - neutro_comp

        if two_variables:
            neutro_comp2 = CompositeSimple_obs(original_data=data2, index=neutro,
                                           mmin=mmin, mmax=mmax)
            data_comp2 = CompositeSimple_obs(original_data=data2, index=case,
                                         mmin=mmin, mmax=mmax)

            comp2 = data_comp2 - neutro_comp2
        else:
            comp2 = None
    except:
        print('Error en ' + s + c)

    if two_variables:
        return comp, case_num, comp2
    else:
        return comp, case_num, data_comp
# ---------------------------------------------------------------------------- #
seasons = ['SON']
min_max_months = [9,11]

cases =  ['DMI_pos', 'DMI_neg', 'N34_pos', 'N34_neg']
cases_cfs = ['dmi_pos', 'dmi_neg', 'n34_pos', 'n34_neg']

variables_cfsv2 = ['prec', 'tref']
variables_obs = ['ppgpcc_w_c_d_1_SON', 'tcru_w_c_d_0.25_SON']

for v_cfsv2, v_obs in zip(variables_cfsv2, variables_obs):

    logger.info(f'{v_cfsv2} : {v_obs}')

    if v_cfsv2 == 'prec':
        fix_mul = 30
    else:
        fix_mul = 1

    logger.info('CFSv2 events')
    # ENSO_CFSv2 ------------------------------------------------------------- #
    logger.info('ENSO')
    en_cfs = xr.open_dataset(f'{cases_dir}{v_cfsv2}_n34_pos_SON_'
                             f'no_detrend_05.nc').rename({v_cfsv2: 'var'})
    en_cfs = en_cfs * fix_mul

    ln_cfs = xr.open_dataset(f'{cases_dir}{v_cfsv2}_n34_neg_SON_'
                             f'no_detrend_05.nc').rename({v_cfsv2: 'var'})
    ln_cfs = ln_cfs * fix_mul

    ln_cfs_clean = ln_cfs.dropna('time')
    en_cfs_clean = en_cfs.dropna('time')

    enso_cfs_test = ttest_ind(en_cfs_clean['var'].values,
                              ln_cfs_clean['var'].values,
                              equal_var=False)[1]

    enso_dif_cfs = en_cfs.mean('time') - ln_cfs.mean('time')

    # IOD_CFSv2 -------------------------------------------------------------- #
    logger.info('IOD')
    iodp_cfs = xr.open_dataset(f'{cases_dir}{v_cfsv2}_dmi_pos_SON_'
                             f'no_detrend_05.nc').rename({v_cfsv2: 'var'})
    iodp_cfs = iodp_cfs * fix_mul

    iodn_cfs = xr.open_dataset(f'{cases_dir}{v_cfsv2}_dmi_neg_SON_'
                             f'no_detrend_05.nc').rename({v_cfsv2: 'var'})
    iodn_cfs = iodn_cfs * fix_mul

    iodn_cfs_clean = iodn_cfs.dropna('time')
    iodp_cfs_clean = iodp_cfs.dropna('time')

    iod_cfs_test = ttest_ind(iodp_cfs_clean['var'].values,
                             iodn_cfs_clean['var'].values,
                             equal_var=False)[1]

    iod_dif_cfs = iodp_cfs.mean('time') - iodn_cfs.mean('time')

    logger.info('Obs events')
    data = xr.open_dataset(f'{data_dir_t_pp}{v_obs}.nc')

    # ENSO_obs --------------------------------------------------------------- #
    logger.info('ENSO obs')
    x, y, en_obs = CaseComp_obs(data, 'SON', mmonth=[9, 11], c='N34_pos',
                                two_variables=False)
    x, y, ln_obs = CaseComp_obs(data, 'SON', mmonth=[9, 11], c='N34_neg',
                                two_variables=False)

    if len(en_obs.sel(lat=slice(20, -60)).lat)>0:
        en_obs = en_obs.sel(lat=slice(20, -60), lon=slice(275, 330))
        ln_obs = ln_obs.sel(lat=slice(20, -60), lon=slice(275, 330))
    else:
        en_obs = en_obs.sel(lat=slice(-60, 20), lon=slice(275, 330))
        ln_obs = ln_obs.sel(lat=slice(-60, 20), lon=slice(275, 330))

    enso_obs_test = ttest_ind(
        en_obs['var'].values, ln_obs['var'].values, equal_var=False)[1]

    enso_dif_obs = en_obs.mean('time') - ln_obs.mean('time')

    # IOD_obs ---------------------------------------------------------------- #
    logger.info('IOD obs')
    x, y, iodp_obs = CaseComp_obs(data, 'SON', mmonth=[9, 11], c='DMI_pos',
                                  two_variables=False)
    x, y, iodn_obs = CaseComp_obs(data, 'SON', mmonth=[9, 11], c='DMI_neg',
                                  two_variables=False)

    if len(iodp_obs.sel(lat=slice(20, -60)).lat)>0:
        iodp_obs = iodp_obs.sel(lat=slice(20, -60), lon=slice(275, 330))
        iodn_obs = iodn_obs.sel(lat=slice(20, -60), lon=slice(275, 330))
    else:
        iodp_obs = iodp_obs.sel(lat=slice(-60, 20), lon=slice(275, 330))
        iodn_obs = iodn_obs.sel(lat=slice(-60, 20), lon=slice(275, 330))

    iod_obs_test = ttest_ind(
        iodp_obs['var'].values, iodn_obs['var'].values, equal_var=False)[1]

    iod_dif_obs = iodp_obs.mean('time') - iodn_obs.mean('time')

    if save is True:
        logger.info('Saving...')
        xr.DataArray(enso_cfs_test).to_netcdf(
            f'{out_dir}{v_cfsv2}_enso_cfsv2_test.nc')
        enso_dif_cfs.to_netcdf(f'{out_dir}{v_cfsv2}_enso_cfsv2_dif.nc')

        xr.DataArray(iod_cfs_test).to_netcdf(
            f'{out_dir}{v_cfsv2}_iod_cfsv2_test.nc')
        iod_dif_cfs.to_netcdf(f'{out_dir}{v_cfsv2}_iod_cfsv2_dif.nc')

        xr.DataArray(enso_obs_test).to_netcdf(
            f'{out_dir}{v_cfsv2}_enso_obs_test.nc')
        enso_dif_obs.to_netcdf(f'{out_dir}{v_cfsv2}_enso_obs_dif.nc')

        xr.DataArray(iod_obs_test).to_netcdf(
            f'{out_dir}{v_cfsv2}_iod_obs_test.nc')
        iod_dif_obs.to_netcdf(f'{out_dir}{v_cfsv2}_iod_obs_dif.nc')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #