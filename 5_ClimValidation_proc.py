'''
Validacion climatologica CFSv2 SON con
Periodo: 1981-2020
Con y sin tendencia
'''
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/val_clim_cfsv2/'
out_dir_proc = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/'

variables = ['prec', 'tref']
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from scipy.stats import ttest_ind
from funciones.selectnmmefiles_utils import SelectNMMEFiles
from funciones.general_utils import ChangeLons, Weights, init_logger
from funciones.preselect_utils import SetDataCFSv2, SplitFilesByMonotonicity

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

import warnings
warnings.filterwarnings('ignore')

# Funciones ------------------------------------------------------------------ #
def fix_calendar(ds, timevar='time'):
    '''
    agrega los dias a los archivos nc de NMME
    '''
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'

    return ds

def CheckFiles(dir, files):
    compute = not all(os.path.isfile(os.path.join(dir, f)) for f in files)
    return compute

# ---------------------------------------------------------------------------- #
logger = init_logger('5_ClimValidation_prec.log')

# ---------------------------------------------------------------------------- #
for v in variables:

    logger.info(f'Variable {v}')
    # ------------------------------------------------------------------------ #
    files = [f'hindcast_{v}_cfsv2_mc_norm_son.nc',
             f'hindcast_{v}_cfsv2_mc_no-norm_son.nc',
             f'hindcast_{v}_cfsv2_mc_detrend_son.nc',
             f'real_time_{v}_cfsv2_mc_norm_son.nc',
             f'real_time_{v}_cfsv2_mc_no-norm_son.nc',
             f'real_time_{v}_cfsv2_mc_detrend_son.nc']

    compute = CheckFiles(out_dir, files)

    # ------------------------------------------------------------------------ #
    if v == 'hgt':
        dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
        dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
    else:
        dir_hc = '/pikachu/datos/osman/nmme/monthly/hindcast/'
        dir_rt = '/pikachu/datos/osman/nmme/monthly/real_time/'

    # ------------------------------------------------------------------------ #
    if compute is True:

        logger.info('Compute')
        logger.info('Procesando hindcast...')

        files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                dir=dir_hc, All=True)
        files = sorted(files, key=lambda x: x.split()[0])
        files = [x for x in files if all(
            year not in x for year in
            ['_2021', '_2022', '_2023', '_2024', '_2025'])]

        # Open files --------------------------------------------------------- #
        try:
            data = xr.open_mfdataset(files, decode_times=False)
            data = SetDataCFSv2(data)

        except Exception as e:
            logger.warning('Error en la monotonía de la dimensión S')
            logger.warning('Usando SplitFilesByMonotonicity...')
            logger.warning(f'Error: {e}')

            files0, files1 = SplitFilesByMonotonicity(files)

            data0 = xr.open_mfdataset(files0, decode_times=False)
            data1 = xr.open_mfdataset(files1, decode_times=False)

            data0 = SetDataCFSv2(data0)
            data1 = SetDataCFSv2(data1)

            data = xr.concat([data0, data1], dim='time')

        logger.info('Hindcast files opened successfully')
        # -------------------------------------------------------------------- #

        # media movil de 3 meses para separar en estaciones
        hindcast = data.rolling(time=3, center=True).mean()
        hindcast = hindcast.load()

        # media climatologica hindcast sin filtrar tendencia
        hindcast_norm = (hindcast.mean(['r', 'time', 'L']) /
                         hindcast.std(['r', 'time', 'L']))
        hindcast_no_norm = hindcast.mean(['r', 'time', 'L'])

        # Save
        hindcast_norm.to_netcdf(f'{out_dir}hindcast_{v}_cfsv2_mc_norm_son.nc')
        hindcast_no_norm.to_netcdf(
            f'{out_dir}hindcast_{v}_cfsv2_mc_no-norm_son.nc')

        print('Detrend...')
        # dos climatologias
        data_1982_1998 = hindcast.sel(
            time=hindcast.time.dt.year.isin(np.linspace(1982, 1998, 17)))
        data_1999_2011 = hindcast.sel(
            time=hindcast.time.dt.year.isin(np.linspace(1999, 2011, 13)))

        for l in [0, 1, 2, 3]:
            # 1982_1998 ------------------------------------------------------ #
            season_1982_1998 = data_1982_1998.sel(
                time=data_1982_1998.time.dt.month.isin(10 - l), L=l)
            # tendencia
            aux = season_1982_1998.mean('r').polyfit(dim='time', deg=1)
            aux_trend = xr.polyval(season_1982_1998['time'],
                                   aux[list(aux.data_vars)[0]])

            if l == 0:
                season_1982_1998_detrened = season_1982_1998 - aux_trend
            else:
                aux_detrend = season_1982_1998 - aux_trend

                season_1982_1998_detrened = \
                    xr.concat([season_1982_1998_detrened,
                               aux_detrend], dim='time')

            # 1999-2011 ------------------------------------------------------ #
            season_1999_2011 = data_1999_2011.sel(
                time=data_1999_2011.time.dt.month.isin(10 - l), L=l)
            # tendencia
            aux = season_1999_2011.mean('r').polyfit(dim='time', deg=1)
            aux_trend = xr.polyval(season_1999_2011['time'],
                                   aux[list(aux.data_vars)[0]])

            if l == 0:
                season_1999_2011_detrend = season_1999_2011 - aux_trend
                season_clim_1999_2011 = season_1999_2011.mean(['r', 'time'])
            else:
                aux_detrend = season_1999_2011 - aux_trend

                season_1999_2011_detrend = \
                    xr.concat([season_1999_2011_detrend,
                               aux_detrend], dim='time')

                season_clim_1999_2011 = \
                    xr.concat([season_clim_1999_2011,
                               season_1999_2011.mean(['r', 'time'])], dim='L')

            son_hindcast_detrend = xr.concat([season_1982_1998_detrened,
                                              season_1999_2011_detrend],
                                             dim='time')
            #son_hindcast_detrend_mean = son_hindcast_detrend.mean(['r', 'time'])

            # Save
            son_hindcast_detrend.to_netcdf(
                f'{out_dir}hindcast_{v}_cfsv2_mc_detrend_son.nc')

        # -------------------------------------------------------------------- #
        logger.info('procesando realtime...')
        files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                                dir=dir_rt, All=True)
        files = sorted(files, key=lambda x: x.split()[0])
        files = [x for x in files if all(
            year not in x for year in
            ['_2021', '_2022', '_2023', '_2024', '_2025'])]

        # Open files --------------------------------------------------------- #
        try:
            data = xr.open_mfdataset(files, decode_times=False)
            data = SetDataCFSv2(data)

        except Exception as e:
            logger.warning('Error en la monotonía de la dimensión S')
            logger.warning('Usando SplitFilesByMonotonicity...')
            logger.warning(f'Error: {e}')

            files0, files1 = SplitFilesByMonotonicity(files)

            if len(np.intersect1d(files0, files1)) == 0:

                # Sin embargo, S parece estar duplicada
                data0 = xr.open_mfdataset(files0, decode_times=False)
                data1 = xr.open_mfdataset(files1, decode_times=False)

                data0 = SetDataCFSv2(data0)
                data1 = SetDataCFSv2(data1)

                t_duplicados = np.intersect1d(data0.time.values, data1.time.values)
                no_duplicados = ~np.isin(data0.time.values, t_duplicados)
                data0 = data0.sel(time=no_duplicados)

                data = xr.concat([data0, data1], dim='time')
            else:
                logger.info('Archivos duplicados en la selecciones de RealTime')

        logger.info('Realtime files opened successfully')
        # -------------------------------------------------------------------- #
        # media movil de 3 meses para separar en estaciones
        real_time = data.rolling(time=3, center=True).mean()
        real_time = real_time.load()

        real_time_norm = (real_time.mean(['r', 'time', 'L']) /
                      real_time.std(['r', 'time', 'L']))

        real_time_no_norm = real_time.mean(['r', 'time', 'L'])

        # Save
        real_time_norm.to_netcdf(f'{out_dir}real_time_{v}_cfsv2_mc_norm_son.nc')
        real_time_no_norm.to_netcdf(
            f'{out_dir}real_time_{v}_cfsv2_mc_no-norm_son.nc')
        del data

        print('Detrend...')
        # la climatologia usada es la de hindcast 1998-2011
        for l in [0, 1, 2, 3]:
            season_data = real_time.sel(
                time=real_time.time.dt.month.isin(10 - l), L=l)
            aux_clim_1999_2011 = season_clim_1999_2011.sel(L=l)

            # Anomalia
            season_anom = season_data - aux_clim_1999_2011

            # Detrend
            aux = season_anom.mean('r').polyfit(dim='time', deg=1)
            aux_trend = xr.polyval(season_anom['time'],
                                   aux[list(aux.data_vars)[0]])

            if l == 0:
                son_realtime_detrend = season_anom - aux_trend
            else:
                aux_detrend = season_anom - aux_trend
                son_realtime_detrend = xr.concat([son_realtime_detrend,
                                                  aux_detrend], dim='time')

        son_realtime_detrend_mean = son_realtime_detrend.mean(['r', 'time'])

        # Save
        son_realtime_detrend.to_netcdf(
            f'{out_dir}real_time_{v}_cfsv2_mc_detrend_son.nc')

    else:
        hindcast_norm = xr.open_mfdataset(
            f'{out_dir}hindcast_{v}_cfsv2_mc_norm_son.nc')
        hindcast_no_norm = xr.open_mfdataset(
            f'{out_dir}hindcast_{v}_cfsv2_mc_no-norm_son.nc')
        son_hindcast_detrend = xr.open_mfdataset(
            f'{out_dir}hindcast_{v}_cfsv2_mc_detrend_son.nc')

        real_time_norm = xr.open_mfdataset(
            f'{out_dir}real_time_{v}_cfsv2_mc_norm_son.nc')
        real_time_no_norm = xr.open_mfdataset(
            f'{out_dir}real_time_{v}_cfsv2_mc_no-norm_son.nc')
        son_realtime_detrend = xr.open_mfdataset(
            f'{out_dir}real_time_{v}_cfsv2_mc_detrend_son.nc')


    logger.info('Observed data...')
    if v == 'prec':
        logger.info('GPCC...')
        data_dir_pp_clim = ('/pikachu/datos/luciano.andrian/observado/ncfiles/'
                            'data_no_detrend/')
        data = xr.open_dataset(
            data_dir_pp_clim + 'pp_pgcc_v2020_1891-2023_1.nc')
        data = data.sel(lat=slice(20, -80), lon=slice(275, 331),
                        time=data.time.dt.year.isin(range(1982, 2020)))

        data = data.interp(lat=hindcast_norm.lat.values,
                           lon=hindcast_norm.lon.values)

        data = data.rolling(time=3, center=True).mean()
        data = data.sel(time=data.time.dt.month.isin(10))  # son

        data = data.rename({'precip': 'prec'})
        data_clim = data.mean('time')
        del data

        # sin tendencia
        data_dir_pp = ('/pikachu/datos/luciano.andrian/observado/ncfiles/'
                       'data_obs_d_w_c/')
        data = xr.open_dataset(data_dir_pp + 'ppgpcc_w_c_d_1_SON.nc')
        data = data.sel(time=data.time.dt.year.isin(range(1981, 2021)),
                        lon=slice(275, 331))
        data = data.rename({'var': 'prec'})
        data_anom = data.interp(lat=hindcast_norm.lat.values,
                           lon=hindcast_norm.lon.values)
        del data

        fix = 30
    elif v=='tref' or v=='T0':

        logger.info('Tcru...')
        data_dir_pp_clim = ('/pikachu/datos/luciano.andrian/observado/ncfiles/'
                            'data_no_detrend/')
        data = xr.open_dataset(data_dir_pp_clim + 't_cru0.25.nc')
        data = ChangeLons(data)
        data = data.sel(lon=slice(275, 331), lat=slice(-70, 20),
                        time=data.time.dt.year.isin(range(1982, 2020)))
        data = data.interp(lat=hindcast_norm.lat.values,
                           lon=hindcast_norm.lon.values)
        data = data.sel(time=data.time.dt.month.isin(10))  # son
        data = data.drop('stn')
        data = data.rename({'tmp': 'tref'})
        data_clim = data.mean('time')
        del data

        # sin tendencia
        data_dir_pp = ('/pikachu/datos/luciano.andrian/observado/ncfiles/'
                       'data_obs_d_w_c/')
        data = xr.open_dataset(data_dir_pp + 'tcru_w_c_d_0.25_SON.nc')
        data = data.sel(lon=slice(275, 331), lat=slice(-70, 20),
                        time=data.time.dt.year.isin(range(1982, 2020)))
        data = data.interp(lat=hindcast_norm.lat.values,
                           lon=hindcast_norm.lon.values)
        data_anom = data.rename({'var': 'tref'})

        del data

    else:
        fix = 1
        pass

    # aux_hind = son_hindcast_detrend + hindcast_norm
    # aux_real = son_realtime_detrend + real_time_norm
    # cfsv2_norm = xr.concat([aux_hind, aux_real], dim='time')

    aux_hind_no_norm = son_hindcast_detrend + hindcast_no_norm
    aux_real_no_norm = son_realtime_detrend + real_time_no_norm
    if v == 'prec':
        cfsv2_no_norm = xr.concat([aux_hind_no_norm,
                                   aux_real_no_norm], dim='time') * fix
    elif v == 'tref':
        cfsv2_no_norm = xr.concat([aux_hind_no_norm,
                                   aux_real_no_norm], dim='time') - 273

    variable_obs = data_anom + data_clim
    # variable_obs_norm = variable_obs/variable_obs.std('time')

    # dif_norm = cfsv2_norm.mean(['time', 'r']) - variable_obs_norm.mean('time')
    # dif_norm = Weights(dif_norm)

    logger.info('Test...')

    dif_no_norm = cfsv2_no_norm.mean(['time', 'r']) - variable_obs.mean('time')
    dif_no_norm = Weights(dif_no_norm)

    # test
    name_var = list(variable_obs.data_vars)[0]
    pvalue = []
    for m in [7, 8, 9, 10]:
        cfsv2_monthly_mean = cfsv2_no_norm.sel(
            time=cfsv2_no_norm.time.dt.month.isin(m)).mean('r')[name_var]
        data_var = variable_obs[name_var]

        # test
        pvalue.append(
            ttest_ind(cfsv2_monthly_mean, data_var, equal_var=False)[1])

    # promedio de pvalue por leadtime
    pvalue = sum(pvalue) / len(pvalue)

    # Save proc
    dif_no_norm.to_netcdf(f'{out_dir_proc}{v}_dif_clim_no-norm.nc')
    xr.DataArray(pvalue).to_netcdf(f'{out_dir_proc}{v}_pvalue_clim_no-norm.nc')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #