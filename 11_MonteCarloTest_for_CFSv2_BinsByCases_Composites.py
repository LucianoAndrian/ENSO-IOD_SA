"""
Igual 10_MonteCarloTest_for_CFSv2_BinsByCases_Composites.py
pero para BinsByCases
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
from funciones.binsbycases_utils import SetBinsByCases, BinsByCases_noComp
from funciones.general_utils import init_logger
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import gc
import multiprocessing as mp

# ---------------------------------------------------------------------------- #
logger = init_logger('11_MonteCarloTest_for_BinsByCases_Composite.log')


# ---------------------------------------------------------------------------- #
dir_events = '/pikachu/datos/luciano.andrian/cases_fields/'

# Functions ------------------------------------------------------------------ #
def SetXr_fromBinsByCases(base, data_to_set, v_name):

    event = data_to_set.dropna(dim='time', how='all')
    proto_event = base.copy().drop_dims('time')
    proto_event[v_name] = event
    proto_event = proto_event.rename({'time':'position'})
    proto_event['position'] = np.arange(len(proto_event.position))

    return proto_event

# Set Composites by cases ---------------------------------------------------- #
dates_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
cases_dir = '/pikachu/datos/luciano.andrian/cases_fields/'

cases_aux = ['sim_pos', 'sim_neg', 'dmi_puros_pos', 'dmi_puros_neg',
             'n34_puros_pos', 'n34_puros_neg']

bin_limits = [[-4.5,-1], #0 s
              [-1, -0.5], #1 m
              [-0.5, 0.5], #2 -
              [0.5, 1], #3  m
              [1, 4.5]] #4 s

indices = ['n34', 'dmi']
magnitudes = ['s', 'm']
cases_names, cases_magnitude, bins_by_cases_n34, bins_by_cases_dmi = \
    SetBinsByCases(indices, magnitudes, bin_limits, cases)

for v in variables:
    logger.info(v)
    if v == 'prec':
        fix = 1 # probando con valores sin fix
        fix_clim = 0
        v_in_name = v
        check_no_t_pp = False
    elif v=='tref':
        fix = 1
        fix_clim = 273
        v_in_name = v
        check_no_t_pp = False
    else:
        fix = 9.8
        fix_clim = 0
        check_no_t_pp = True
        if v == 'hgt750':
            v_in_name = 'HGT'
        else:
            v_in_name = 'hgt'

    aux_comps = {}
    aux_num_comps = {}
    n_count = 0

    logger.info('comp...')
    for c_count, c in enumerate(cases):
        logger.info(f'case: {c}')
        cases_bin, num_bin, auxx = BinsByCases_noComp(
            v=v, v_name=v_in_name, fix_factor=fix, s='SON', mm=10, c=c,
            c_count=c_count,
            bin_limits=bin_limits,
            bins_by_cases_dmi=bins_by_cases_dmi,
            bins_by_cases_n34=bins_by_cases_n34,
            snr=False, cases_dir=cases_dir, dates_dir=dates_dir,
            neutro_clim=True)

        bins_aux_dmi = bins_by_cases_dmi[c_count]
        bins_aux_n34 = bins_by_cases_n34[c_count]

        for b_n34 in range(0, len(bins_aux_n34)):
            for b_dmi in range(0, len(bins_aux_dmi)):
                aux_comps[cases_names[n_count]] = \
                    SetXr_fromBinsByCases(auxx, cases_bin[b_dmi][b_n34], v)
                aux_num_comps[cases_names[n_count]] = num_bin[b_dmi][b_n34]
                n_count += 1

    if v == 'hgt':
        end_name_file = '_05'
    elif v == 'hgt750':
        end_name_file = '__detrend_05' # aaaah lrpm
    else:
        end_name_file = '_detrend_05'

    neutro = xr.open_dataset(f'{dir_events}{v}_neutros_SON{end_name_file}.nc')
    len_neutro = len(neutro.time)
    neutro = neutro.rename({'time': 'position'})
    neutro = neutro.drop_vars(['r', 'L'])
    if v == 'hgt750':
        neutro = neutro.rename({'HGT': 'hgt750'})
        neutro = neutro.sel(P=750)

    logger.info('MC...')
    for k in aux_comps.keys(): # son los "cases" de cada categoria
        logger.info(k)
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
        event = aux_comps[k].copy()

        concat = xr.concat([neutro, event], dim='position')

        concat['position'] = np.arange(len(concat.position))

        if check_no_t_pp is True:
            aux = concat.sel(lat=np.arange(-60, 20 + 1))
            if len(aux.lat.values)==0:
                aux = concat.sel(lat=np.arange(20 + 1, -60))
            concat = aux.sel(lon=np.arange(270, 330 + 1))

        if v == 'hgt750':
            concat = concat.drop_vars('P')

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

        hour = datetime.now().hour

        if (hour > 19) | (hour < 8):
            n_proc = 20
        else:
            n_proc = 10

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

        logger.info('Quantiles')
        aux = aux.chunk({'position': -1})
        qt = aux.quantile([.05, .95], dim='position',
                          interpolation='linear')
        qt.to_netcdf(f'{out_dir}{v}_QT_{k}_CFSv2_detrend_05.nc',
                     compute=True)
        aux.close()
        del qt, aux
        gc.collect()

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #