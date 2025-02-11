"""
Pre-procesamiento hgt200, prec, tref, tsigma095
Anomalías respecto a la climatologia del hindcast y detrend de las anomalias
(similar 2_fixCFSv2_DMI_N34.py)
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from Funciones import SelectNMMEFiles
# ---------------------------------------------------------------------------- #
out_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
save_nc = False
variables = ['hgt', 'tref', 'prec', 'T0995sigma']

# Funciones ------------------------------------------------------------------ #
def fix_calendar(ds, timevar='time'):
    """
    agrega los dias a los archivos nc de NMME
    """
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'
    return ds

def TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, main_month_season):

    for l in [0,1,2,3]:
        season_1982_1998 = \
            data_1982_1998.sel(
                time=data_1982_1998.time.dt.month.isin(main_month_season-l),
                L=l)
        season_1999_2011 = \
            data_1999_2011.sel(
                time=data_1999_2011.time.dt.month.isin(main_month_season-l),
                L=l)

        if l==0:
            season_clim_1982_1998 = season_1982_1998.mean(['r', 'time'])
            season_clim_1999_2011 = season_1999_2011.mean(['r', 'time'])

            season_anom_1982_1998 = season_1982_1998 - season_clim_1982_1998
            season_anom_1999_2011 = season_1999_2011 - season_clim_1999_2011
        else:
            season_clim_1982_1998 = \
                xr.concat([season_clim_1982_1998,
                           season_1982_1998.mean(['r', 'time'])], dim='L')
            season_clim_1999_2011 = \
                xr.concat([season_clim_1999_2011,
                           season_1999_2011.mean(['r', 'time'])], dim='L')

            aux_1982_1998 = \
                season_1982_1998 - season_1982_1998.mean(['r', 'time'])
            aux_1999_2011 = \
                season_1999_2011 - season_1999_2011.mean(['r', 'time'])

            season_anom_1982_1998 = \
                xr.concat([season_anom_1982_1998, aux_1982_1998], dim='time')
            season_anom_1999_2011 =\
                xr.concat([season_anom_1999_2011, aux_1999_2011], dim='time')

    return season_clim_1982_1998, season_clim_1999_2011,\
           season_anom_1982_1998, season_anom_1999_2011

def Anom_SeasonRealTime(data_realtime, season_clim_1999_2011,
                                main_month_season):

    for l in [0,1,2,3]:
        season_data = data_realtime.sel(
            time=data_realtime.time.dt.month.isin(main_month_season-l), L=l)
        aux_season_clim_1999_2011 = season_clim_1999_2011.sel(L=l)

        #Anomalia
        season_anom = season_data - aux_season_clim_1999_2011

        if l==0:
            season_anom_f = season_anom
        else:
            season_anom_f = xr.concat([season_anom_f, season_anom], dim='time')

    return season_anom_f

# ---------------------------------------------------------------------------- #
for v in variables:
    print(f"{v} ------------------------------------------------------------ #")

    if v == 'T0995sigma' or v == 'hgt':
        dir_hc = '/pikachu/datos/luciano.andrian/hindcast/'
        dir_rt = '/pikachu/datos/luciano.andrian/real_time/'
    else:
        dir_hc = '/pikachu/datos/osman/nmme/monthly/hindcast/'
        dir_rt = '/pikachu/datos/osman/nmme/monthly/real_time/'
    # usando SelectNMMEFiles con All=True,
    # abre TODOS los archivos .nc de la ruta en dir

    # HINDCAST -----------------------------------------------------------------
    print('hindcast')
    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                            dir=dir_hc, All=True)
    files = sorted(files, key=lambda x: x.split()[0])

    # abriendo todos los archivos
    # xr no entiende la codificacion de Leads, r y las fechas
    data = xr.open_mfdataset(files, decode_times=False)
    data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
    data = data.sel(L=[0.5, 1.5, 2.5, 3.5])  # Solo leads 0 1 2 3
    data['L'] = [0, 1, 2, 3]
    data = xr.decode_cf(fix_calendar(data))  # corrigiendo fechas
    data = data.sel(lon=slice(275, 330))
    if v == 'T0995sigma':
        data = data.sel(lat=slice(15, -60))
    else:
        data = data.sel(lat=slice(-60, 15))
    data = data.sel(r=slice(1, 24))

    # media movil de 3 meses para separar en estaciones
    data = data.rolling(time=3, center=True).mean()

    # 1982-1998, 1999-2011
    data_1982_1998 = \
        data.sel(time=data.time.dt.year.isin(np.linspace(1982, 1998, 17)))
    data_1999_2011 = \
        data.sel(time=data.time.dt.year.isin(np.linspace(1999, 2011, 13)))

    # - Climatologias y anomalias detrend por seasons --------------------------
    # --------------------------------------------------------------------------
    son_clim_82_98, son_clim_99_11, son_anom_82_98, son_anom_99_11 = \
        TwoClim_Anom_Seasons(data_1982_1998, data_1999_2011, 10)

    son_hindcast = xr.concat([son_anom_82_98, son_anom_99_11], dim='time')

    son_hindcast.to_netcdf(f"{out_dir}{v}_aux_hindcast_no_detrend_son.nc")
    son_clim_99_11.to_netcdf(f"{out_dir}{v}_aux_son_clim_99_11.nc")
    del son_hindcast, son_clim_82_98, son_clim_99_11

    # --------------------------------------------------------------------------
    # Real-time ----------------------------------------------------------------
    print('real-time')
    files = SelectNMMEFiles(model_name='NCEP-CFSv2', variable=v,
                            dir=dir_rt, All=True)
    files = sorted(files, key=lambda x: x.split()[0])
    files = [x for x in files if "_2022" not in x and '_2021' not in x]

    if v == 'tref':
        # para evitar: ValueError:
        # Resulting object does not have monotonic global indexes along
        # dimension
        # en xr.open_mfdataset
        files0 = files[0:144]
        files1 = files[145:len(files)]

        data0 = xr.open_mfdataset(files0, decode_times=False).sel(
            L=[0.5, 1.5, 2.5, 3.5], M=slice(1, 24), X=slice(275, 331),
            Y=slice(-70, 20))
        data1 = xr.open_mfdataset(files1, decode_times=False).sel(
            L=[0.5, 1.5, 2.5, 3.5], M=slice(1, 24), X=slice(275, 331),
            Y=slice(-70, 20))

        data0 = data0.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
        data0['L'] = [0, 1, 2, 3]
        data0 = xr.decode_cf(fix_calendar(data0))  # corrigiendo fechas
        data0 = data0.sel(time=data0.time.dt.year.isin(2011))
        data0 = data0.sel(r=slice(1, 24))

        data1 = data1.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
        data1['L'] = [0, 1, 2, 3]
        data1 = xr.decode_cf(fix_calendar(data1))  # corrigiendo fechas
        data1 = data1.sel(
            time=data1.time.dt.year.isin(np.linspace(2012, 2020, 9)))
        data1 = data1.sel(r=slice(1, 24))

        data = xr.concat([data0, data1], dim='time')
    else:
        # xr no entiende la codificacion de Leads, r y las fechas
        data = xr.open_mfdataset(files, decode_times=False)
        data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
        data = data.sel(L=[0.5, 1.5, 2.5, 3.5])  # Solo leads 0 1 2 3
        data['L'] = [0, 1, 2, 3]
        data = xr.decode_cf(fix_calendar(data))  # corrigiendo fechas
        data = data.sel(
            time=data.time.dt.year.isin(np.linspace(2011, 2020, 10)))
        data = data.sel(lon=slice(275, 330))
        if v == 'T0995sigma':
            data = data.sel(lat=slice(15, -60))
        else:
            data = data.sel(lat=slice(-60, 15))
        data = data.sel(r=slice(1, 24))

    # data = xr.open_mfdataset(files, decode_times=False).sel(
    #     L=[0.5, 1.5, 2.5, 3.5], M=slice(1,24), Y=slice(-70, 20))
    # data['L'] = [0,1,2,3]
    # data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
    # data = xr.decode_cf(fix_calendar(data))

    data = data.rolling(time=3, center=True).mean()

    # - Anomalias detrend por seasons ------------------------------------------
    # --------------------------------------------------------------------------
    son_clim_99_11 = xr.open_dataset(f"{out_dir}{v}_aux_son_clim_99_11.nc")
    son_hindcast_no_detrend = \
        xr.open_dataset(f"{out_dir}{v}_aux_hindcast_no_detrend_son.nc")

    son_realtime_no_detrend = Anom_SeasonRealTime(data, son_clim_99_11, 10)
    son_realtime_no_detrend.load()

    print('concat')
    son_f = xr.concat(
        [son_hindcast_no_detrend, son_realtime_no_detrend], dim='time')

    # save ---------------------------------------------------------------------
    if save_nc:
        son_f.to_netcdf(f"{out_dir}{v}_son_no_detrend.nc")

    del son_realtime_no_detrend, son_hindcast_no_detrend, data, son_f
print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')