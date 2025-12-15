"""
Test de Monte Carlo para las composiciones en CFSv2
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = ('/pikachu/datos/luciano.andrian/observado/ncfiles/'
           'CFSv2_nc_quantiles/')

variables = ['prec', 'tref', 'hgt', 'hgt750']

cases = ['sim_pos', 'sim_neg',
         'dmi_puros_pos', 'dmi_puros_neg',
         'n34_puros_pos', 'n34_puros_neg']

# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import os
import glob
from datetime import datetime
from funciones.montecarlo_utils import NumberPerts, CompositesSimple_CFSv2
from funciones.general_utils import init_logger
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import gc
# ---------------------------------------------------------------------------- #
logger = init_logger('10_MonteCarloTest_for_CFSv2Composite.log')

# ---------------------------------------------------------------------------- #
dir_events = '/pikachu/datos/luciano.andrian/cases_fields/'

# ---------------------------------------------------------------------------- #
for v in variables:
    logger.info(v)
    if v == 'hgt':
        neutro = xr.open_dataset(f'{dir_events}{v}_neutros_SON_05.nc')
    elif v == 'hgt750':
        neutro = xr.open_dataset(f'{dir_events}{v}_neutros_SON__detrend_05.nc')
    else:
        neutro = xr.open_dataset(f'{dir_events}{v}_neutros_SON_detrend_05.nc')

    if v == 'hgt' or v == 'hgt750':
        if len(neutro.sel(lat=slice(-60, 20)).lat.values) > 0:
            neutro = neutro.sel(lat=slice(-60, 20), lon=slice(275, 330))
        else:
            neutro = neutro.sel(lat=slice(20, -60), lon=slice(275, 330))

    len_neutro = len(neutro.time)
    neutro = neutro.rename({'time': 'position'})
    neutro = neutro.drop_vars(['r', 'L'])

    for c in cases:
        logger.info(f'case: {c}')
        # Borrar archivos temporales ----------------------------------------- #
        files = glob.glob('/pikachu/datos/luciano.andrian/'
                          'observado/ncfiles/nc_comps/' + '*.nc')

        if len(files) != 0:
            for f in files:
                try:
                    os.remove(f)
                except:
                    print('Error: ' + f)

        # -------------------------------------------------------------------- #
        if v == 'hgt':
            event = xr.open_dataset(f'{dir_events}{v}_{c}_SON_05.nc')
        elif v == 'hgt750':
            event = xr.open_dataset(f'{dir_events}{v}_{c}_SON__detrend_05.nc')
        else:
            event = xr.open_dataset(f'{dir_events}{v}_{c}_SON_detrend_05.nc')

        if v == 'hgt' or v == 'hgt750':
            if len(event.sel(lat=slice(-60, 20)).lat.values) > 0:
                event = event.sel(lat=slice(-60, 20), lon=slice(275, 330))
            else:
                event = event.sel(lat=slice(20, -60), lon=slice(275, 330))


        event = event.rename({'time': 'position'})
        event = event.drop_vars(['r', 'L'])

        concat = xr.concat([neutro, event], dim='position')

        concat['position'] = np.arange(len(concat.position))

        def PermuDatesComposite(n, data=concat, len_neutro=len_neutro):

            for a in n:
                rn = np.random.RandomState(616 + int(a))
                dates_rn = rn.permutation(data.position)
                neutro_new = dates_rn[0:len_neutro]
                data_new = dates_rn[len_neutro:]

                neutro_comp = CompositesSimple_CFSv2(data, neutro_new)
                event_comp = CompositesSimple_CFSv2(data, data_new)

                comp = event_comp - neutro_comp

                if a == n[0]:
                    comp_concat = comp
                else:
                    comp_concat = xr.concat([comp_concat, comp],
                                            dim='position')

            comp_concat.to_netcdf('/pikachu/datos/luciano.andrian/observado'
                                  '/ncfiles/nc_comps/' + 'Comps_' +
                                  str(int(a)) + '.nc')

            del neutro_comp, event_comp, comp, comp_concat
            gc.collect()

        M = NumberPerts(concat, neutro, 0)
        import multiprocessing as mp
        hour = datetime.now().hour

        if (hour > 19) | (hour < 8):
            n_proc = 30
        else:
            n_proc = 20

        logger.info(f'Procesos: {n_proc}')

        with mp.Pool(processes=n_proc) as pool:
            pool.map(PermuDatesComposite, [n for n in M])
        logger.info('pool ok')
        del event, concat
        gc.collect()

        # NO chunks! acumula RAM en cada ciclo. ~14gb en 3 ciclos...
        aux = xr.open_mfdataset('/pikachu/datos/luciano.andrian/'
                                'observado/ncfiles/'
                                'nc_comps/Comps_*.nc', parallel=True,
                                combine='nested', concat_dim="position",
                                coords="different",
                                compat="broadcast_equals")

        logger.info('Quantiles...')
        aux = aux.chunk({'position': -1})
        qt = aux.quantile([.05, .95], dim='position',
                          interpolation='linear')
        qt.to_netcdf(f'{out_dir}{v}_QT_{c}_CFSv2_detrend_05.nc',
                     compute=True)
        aux.close()
        del qt, aux
        gc.collect()

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #