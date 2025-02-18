"""
Test de Monte Carlo para las composiciones de eventos observados
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/nc_quantiles' \
          '/DMIbase/'

variables = ['HGT200_SON_mer_d_w', 'HGT750_SON_mer_d_w',
             'tcru_w_c_d_0.25_SON', 'ppgpcc_w_c_d_1_SON']

# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from multiprocessing.pool import ThreadPool
import os
import glob
import math
from datetime import datetime
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'

dirs = [data_dir, data_dir, data_dir_t_pp, data_dir_t_pp]

periodos = [['40', '20']]

cases = ['DMI_sim_pos', 'DMI_sim_neg', 'DMI_un_pos',
         'DMI_un_neg', 'N34_un_pos', 'N34_un_neg']

seasons = ['SON']
min_max_months = [[9, 11]]

# Functions ------------------------------------------------------------------ #
def CompositeSimple(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(
            time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(month=comp_field['time.month'], mmin=mmin, mmax=mmax))
        if len(comp_field.time) != 0:
            comp_field = comp_field.mean(['time'], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field
    else:
        print(' len index = 0')

def NumberPerts(data_to_concat, neutro, num = 0):
    #si las permutaciones posibles:
    # > 10000 --> permutaciones = 10000
    # < 1000 ---> permutaciones = todas las posibles
    #
    # si num != 0 permutaciones = num

    total = len(data_to_concat) + len(neutro)
    len1 = len(neutro)
    len2 = len(data_to_concat)

    total_perts = math.factorial(total) / (math.factorial(len2) * \
                                           math.factorial(len1))

    if num == 0:
        if total_perts >= 10000:
            tot = 10000
            print('M = 10000')
        else:
            tot = total_perts
            print('M = ' + str(total_perts))

    else:
        tot = num

    jump = 9 #10 por .nc que guarde
    M = []
    n = 0

    while n < tot:
        aux = list(np.linspace((0 + n), (n + jump), (jump + 1)))
        M.append(aux)
        n = n + jump + 1

    return M

# ---------------------------------------------------------------------------- #
for v, dir in zip(variables, dirs):
    print(v)
    data = xr.open_mfdataset(f'{dir}{v}.nc')

    if len(data.sel(lat=slice(20, -60)).lat)>0:
        data = data.sel(lat=slice(20, -60), lon=slice(275, 330))
        lat_dec=True
    else:
        data = data.sel(lat=slice(-60, 20), lon=slice(275, 330))
        lat_dec = False

    if 'HGT' in v:
        print('hgt...')
        if lat_dec is True:
            lat_interp = np.arange(-60, 20, 0.5)[::-1]
        else:
            lat_interp = np.arange(-60, 20, 0.5)

        data = data.interp(lat=lat_interp,
                           lon=np.arange(275, 330, 0.5))

    print(data.lat.values)

    for c_count, c in enumerate(cases):
        print(f'{c}-----------------------------------------------------------')
        for s_count, s in enumerate(seasons):
            print(s)
            # si la proxima tenga menos de 10000 permutaciones
            # no se sobreescribirian todas
            files = glob.glob('/pikachu/datos/luciano.andrian/'
                              'observado/ncfiles/nc_comps/' + '*.nc')

            if len(files) != 0:
                for f in files:
                    try:
                        os.remove(f)
                    except:
                        print('Error: ' + f)

            mmonth = min_max_months[s_count]

            def PermuDatesComposite(n, data=data, mmonth=mmonth):
                mmin = mmonth[0]
                mmax = mmonth[-1]
                rn = np.random.RandomState(616)

                for a in n:
                    dates_rn = rn.permutation(neutro_concat)
                    neutro_new = dates_rn[0:len(neutro)]
                    data_new = dates_rn[len(neutro):]

                    neutro_comp = CompositeSimple(original_data=data,
                                                  index=neutro_new,
                                                  mmin=mmin, mmax=mmax)
                    data_comp = CompositeSimple(original_data=data,
                                                index=data_new,
                                                mmin=mmin, mmax=mmax)

                    if (len(data_comp) != 0):
                        if a == n[0]:
                            comp = data_comp - neutro_comp
                            comp = comp.expand_dims(time=[a])
                            comp_concat = comp
                        else:
                            comp = data_comp - neutro_comp
                            comp = comp.expand_dims(time=[a])
                            try:
                                comp_concat = xr.concat(
                                    [comp_concat, comp], dim='time')
                            except:
                                if a != n[0]:
                                    comp_concat = comp
                    else:
                        next

                comp_concat.to_netcdf(
                    '/pikachu/datos/luciano.andrian/observado'
                    '/ncfiles/nc_comps/' + 'Comps_' +
                    str(int(a)) + '.nc')

                del comp, data_comp, neutro_new, comp_concat


            # Fechas de los eventos IODS y Ninios detectados a partir
            # de ERSSTv5 en 1920-2020
            aux = xr.open_dataset(nc_date_dir + '1920_2020_' + s + '.nc')

            neutro = aux.Neutral
            data_to_concat = aux[c]
            aux.close()

            M = NumberPerts(data_to_concat, neutro, 0)

            if (data_to_concat[0] != 0):
                neutro_concat = np.concatenate([neutro, data_to_concat])

                hour = datetime.now().hour
                if (hour > 19) | (hour < 8):
                    n_thread = 30
                    pool = ThreadPool(30)
                else:
                    n_thread = 10
                    pool = ThreadPool(10)

                print('Threads: ', n_thread)

                pool.map(PermuDatesComposite, [n for n in M])
                pool.close()

                # NO chunks! acumula RAM en cada ciclo. ~14gb en 3 ciclos...
                aux = xr.open_mfdataset('/pikachu/datos/luciano.andrian/'
                                        'observado/ncfiles/'
                                        'nc_comps/Comps_*.nc', parallel=True,
                                        combine='nested', concat_dim="time",
                                        coords="different",
                                        compat="broadcast_equals")

                print(f'Quantiles ------------------------------------------- ')
                aux = aux.chunk({'time': -1})
                qt = aux.quantile([.05, .95], dim='time',
                                  interpolation='linear')
                qt.to_netcdf(out_dir + v + '_' + c + '1940_2020_' + s + '.nc',
                             compute=True)

                aux.close()
                del qt
            else:
                print('no ' + c)

print('# ------------------------------------------------------------------- #')
print('# ------------------------------------------------------------------- #')
print('done')
print('# ------------------------------------------------------------------- #')