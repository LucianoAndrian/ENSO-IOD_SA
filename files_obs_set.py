"""
Seteo de archivos observados
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir_era = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'
out_dir_p_t = '/pikachu/datos/luciano.andrian/observado/ncfiles/data_obs_d_w_c/'

# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np

# ---------------------------------------------------------------------------- #
dir_files_era = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/'\
                'downloaded/'
dir_files_p_t = '/pikachu/datos/luciano.andrian/observado/ncfiles/'\
                'data_no_detrend/'

# Funciones ------------------------------------------------------------------ #
def Detrend(xrda, dim):
    aux = xrda.polyfit(dim=dim, deg=1)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients)
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients)
    dt = xrda - trend
    return dt

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

def ChangeLons(data, lon_name='lon'):
    data['_longitude_adjusted'] = xr.where(
        data[lon_name] < 0,
        data[lon_name] + 360,
        data[lon_name])

    data = (
        data
            .swap_dims({lon_name: '_longitude_adjusted'})
            .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
            .drop(lon_name))

    data = data.rename({'_longitude_adjusted': 'lon'})

    return data

print('# ERA5 -------------------------------------------------------------- #')
variables =['HGT200', 'HGT750']
name_variables = ['z', 'z']

for v, n_v in zip(variables, name_variables):

    print(v)
    data = xr.open_dataset(f'{dir_files_era}ERA5_{v}_40-20.nc')

    if n_v == 'u':
        print('Drop v')
        data = data.drop('v')
    elif n_v == 'v':
        print('Drop u')
        data = data.drop('u')

    data = data.rename({n_v: 'var'})
    data = data.rename({'longitude': 'lon'})
    data = data.rename({'latitude': 'lat'})

    data = Weights(data)
    data = data.sel(lat=slice(20, -80))
    data = data.rolling(time=3, center=True).mean()

    for mm, s_name in zip([7, 10], ['JJA', 'SON']):
        aux = data.sel(time=data.time.dt.month.isin(mm))
        aux = Detrend(aux, 'time')

        if save is True:
            print('to_netcdf...')
            if v == 'UV200':
                aux.to_netcdf(f'{out_dir_era}{n_v}_{s_name}_mer_d_w.nc')
            else:
                aux.to_netcdf(f'{out_dir_era}{v}_{s_name}_mer_d_w.nc')


print('# GPCC - TCRU ------------------------------------------------------- #')
# T CRU
data = xr.open_dataset(dir_files_p_t + 't_cru0.25.nc')
data = data.drop('stn')
data = data.rename({'tmp':'var'})
data = ChangeLons(data)
data_40_20 = data.sel(time=slice('1940-01-16', '2020-12-16'))
del data

data_40_20 = Weights(data_40_20)
data_40_20 = data_40_20.sel(lat=slice(-80, 20)) # HS
data_40_20 = data_40_20.rolling(time=3, center=True).mean()
for mm, s_name in zip([7,10], ['JJA','SON']):  # en caso de sumar otras...
    aux = data_40_20.sel(time=data_40_20.time.dt.month.isin(mm))
    aux = Detrend(aux, 'time')
    if save is True:
        aux.to_netcdf(out_dir_p_t + 'tcru_w_c_d_0.25_'+ s_name + '.nc')
del data_40_20, aux

# PP GPCC
data = xr.open_dataset(dir_files_p_t + 'pp_pgcc_v2020_1891-2023_1.nc')
data = data.rename({'precip':'var'})
data_40_20 = data.sel(time=slice('1940-01-16', '2020-12-16'))
del data

data_40_20 = Weights(data_40_20)
data_40_20 = data_40_20.sel(lat=slice(20, -80)) # HS
data_40_20 = data_40_20.rolling(time=3, center=True).mean()
for mm, s_name in zip([7, 10], ['JJA','SON']):  # en caso de sumar otras...
    aux = data_40_20.sel(time=data_40_20.time.dt.month.isin(mm))
    aux = Detrend(aux, 'time')
    if save is True:
        aux.to_netcdf(out_dir_p_t + 'ppgpcc_w_c_d_1_'+ s_name + '.nc')

print('# ------------------------------------------------------------------- #')
print('# ------------------------------------------------------------------- #')
print('done')
print('# ------------------------------------------------------------------- #')