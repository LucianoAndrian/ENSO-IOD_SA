'''
Validacion climatologica CFSv2 SON con
Periodo: 1981-2020
Con y sin tendencia
'''
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/pikachu/datos/luciano.andrian/val_clim_cfsv2/'
out_dir_proc = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/'

variables = ['prec', 'tref']#, 'T09955sigma']#, 'hgt']
print('# ------------------------------------------------------------------- #')
print('# Tsigma y hgt no configurado --------------------------------------- #')
print('# ------------------------------------------------------------------- #')
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from Funciones import SelectNMMEFiles, ChangeLons

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# ---------------------------------------------------------------------------- #
if save:
    dpi = 300
else:
    dpi = 100

# Funciones ------------------------------------------------------------------ #
def fix_calendar(ds, timevar='time'):
    '''
    agrega los dias a los archivos nc de NMME
    '''
    if ds[timevar].attrs['calendar'] == '360':
        ds[timevar].attrs['calendar'] = '360_day'

    return ds

def Plot(comp, comp_var, levels, save, dpi, title, name_fig, out_dir, cmap):

    import matplotlib.pyplot as plt
    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    fig_size = (5, 6)
    extent = [270, 330, -60, 20]
    xticks = np.arange(275, 330+1, 10)
    yticks = np.arange(-60, 15+1, 10)
    crs_latlon = ccrs.PlateCarree()

    levels_contour = levels.copy()
    if isinstance(levels, np.ndarray):
        levels_contour = levels[levels != 0]
    else:
        levels_contour.remove(0)

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(extent, crs=crs_latlon)
    im = ax.contourf(comp.lon, comp.lat, comp_var, levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035,shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.BORDERS, facecolor='k')
    ax.add_feature(cartopy.feature.OCEAN, zorder=10, facecolor='white',
                   edgecolor='k')
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', zorder=12)
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    # al usar ocean como mascara
    # y las girdlanes encima
    # los bordes del plot quedan tapadaos por las gridlines
    for k, spine in ax.spines.items():
        spine.set_zorder(13)

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=7)

    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

def CheckFiles(dir, files):
    compute = not all(os.path.isfile(os.path.join(dir, f)) for f in files)
    return compute

def SetDataCFSv2(data):
    data = data.rename({'X': 'lon', 'Y': 'lat', 'M': 'r', 'S': 'time'})
    data = data.sel(L=[0.5, 1.5, 2.5, 3.5], r=slice(1, 24),
                    lon=slice(275, 331), lat=slice(-70, 20))
    data['L'] = [0, 1, 2, 3]
    data = xr.decode_cf(fix_calendar(data))  # corrigiendo fechas

    return data

def SplitFilesByMonotonicity(files):
    """ Divide la lista de archivos en segmentos donde el índice S
    sea monotónico """
    for i in range(1, len(files)):
        try:
            # Intentar abrir los archivos hasta el índice i
            xr.open_mfdataset(files[:i], decode_times=False)
        except ValueError as e:
            if "monotonic global indexes along dimension S" in str(e):
                return files[:i-1], files[i-1:]
    return files, []

# ---------------------------------------------------------------------------- #
for v in variables:

    print(f'Variable {v}')
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

        print('Compute')

        print('# ----------------------------------------------------------- #')
        print('Procesando hindcast...')

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

        except:
            print('Error en la monotonia de la dimencion S')
            print('Usando SplitFilesByMonotonicity...')
            files0, files1 = SplitFilesByMonotonicity(files)

            data0 = xr.open_mfdataset(files0, decode_times=False)
            data1 = xr.open_mfdataset(files1, decode_times=False)

            data0 = SetDataCFSv2(data0)
            data1 = SetDataCFSv2(data1)

            data = xr.concat([data0, data1], dim='time')

        print('Open files done')
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

        print('# ----------------------------------------------------------- #')
        print('procesando realtime...')
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

        except:
            print('Error en la monotonia de la dimencion S')
            print('Usando SplitFilesByMonotonicity...')
            files0, files1 = SplitFilesByMonotonicity(files)

            data0 = xr.open_mfdataset(files0, decode_times=False)
            data1 = xr.open_mfdataset(files1, decode_times=False)

            data0 = SetDataCFSv2(data0)
            data1 = SetDataCFSv2(data1)

            data = xr.concat([data0, data1], dim='time')

        print('Open files done')
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


    if v == 'prec':
        print('GPCC...')
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

        print('Tcru...')
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

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')