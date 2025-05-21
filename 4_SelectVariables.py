"""
Seleccion de los campos de las variables para cada caso de eventos IOD y ENSO
A partir de los sst_* salida de 2_CFSv2_DMI_N34.py ara asegurar
correspondencia entre los eventos de los índices y los campos seleccionados.
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
from multiprocessing.pool import ThreadPool
from multiprocessing import Process
from Funciones import SelectVariables
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# ---------------------------------------------------------------------------- #
save_nc = True
out_dir = '/pikachu/datos/luciano.andrian/cases_fields/'

cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
data_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
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

# SST ##########################################################################
if len(seasons)>1:
    def SelectEvents(c):
        for s in seasons:
            aux_cases = \
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '.nc')\
                    .rename({'__xarray_dataarray_variable__': 'index'})
            data_sst_s = xr.open_dataset(data_dir + 'sst_' + s.lower() + '.nc')
            case_events = SelectVariables(aux_cases, data_sst_s)
            case_events.to_netcdf(out_dir + c + '_' + s + '.nc')


    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEvents, [c for c in cases])
else:
    print('one season')
    def SelectEvents(c):
        s=seasons[0]
        aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
            .rename({'__xarray_dataarray_variable__': 'index'})
        data_sst_s = xr.open_dataset(data_dir + 'sst_' + s.lower() + '.nc')
        case_events = SelectVariables(aux_cases, data_sst_s)
        case_events.to_netcdf(out_dir + c + '_' + s + '_05.nc')

    processes = [Process(target=SelectEvents, args=(c,)) for c in cases]
    for process in processes:
        process.start()

# HGT ##########################################################################
if len(seasons)>1:
    def SelectEventsHGT(c):
        for s in seasons:
            try:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'__xarray_dataarray_variable__': 'index'})
            except:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'sst': 'index'})

            data_hgt_s = xr.open_dataset(data_dir + 'hgt_' + s.lower() + '.nc')
            case_events = SelectVariables(aux_cases, data_hgt_s)

            case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '_05.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsHGT, [c for c in cases])

else:
    print('one season')
    def SelectEventsHGT(c):
        s = seasons[0]
        try:
            aux_cases = \
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                    .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                .rename({'sst': 'index'})

        data_hgt_s = xr.open_dataset(data_dir + 'hgt_' + s.lower() + '.nc')

        case_events = SelectVariables(aux_cases, data_hgt_s)
        case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '_05.nc')


    processes = [Process(target=SelectEventsHGT, args=(c,)) for c in cases]
    for process in processes:
        process.start()

#cases = ['dmi_neg_n34_pos', 'dmi_pos_n34_neg']
# TSigma #######################################################################
# if len(seasons)>1:
#     def SelectEventsTSigma(c):
#         for s in seasons:
#             try:
#                 aux_cases = \
#                     xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc') \
#                     .rename({'__xarray_dataarray_variable__': 'index'})
#             except:
#                 aux_cases = \
#                     xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc') \
#                     .rename({'TMP': 'index'})
#
#             data_tsigma_s =\
#                 xr.open_dataset(data_dir + 'tsigma_' + s.lower() + '.nc')
#             case_events = SelectVariables(aux_cases, data_tsigma_s)
#
#             case_events.to_netcdf(out_dir + 'tsigma_' + c + '_' + s + '_05.nc')
#
#     pool = ThreadPool(4)  # uno por season
#     pool.map_async(SelectEventsTSigma, [c for c in cases])
#
# else:
#     print('one season')
#     def SelectEventsTSigma(c):
#         s = seasons[0]
#         try:
#             aux_cases =\
#                 xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
#                     .rename({'__xarray_dataarray_variable__': 'index'})
#         except:
#             aux_cases = \
#                 xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
#                     .rename({'sst': 'index'})
#
#         data_tsigma_s = \
#             xr.open_dataset(data_dir + 'T0995sigma_' + s.lower() +
#                             '_no_detrend.nc')
#
#         case_events = SelectVariables(aux_cases, data_tsigma_s)
#         case_events.to_netcdf(out_dir + 'tsigma_' + c + '_' + s +
#                               '_no_detrend_05.nc')
#
#     processes = [Process(target=SelectEventsTSigma, args=(c,)) for c in cases]
#     for process in processes:
#         process.start()

# Tref #########################################################################
#cases = ['dmi_neg_n34_pos', 'dmi_pos_n34_neg']
if len(seasons)>1:
    def SelectEventsTref(c):
        for s in seasons:
            try:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc') \
                    .rename({'__xarray_dataarray_variable__': 'index'})
            except:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc') \
                    .rename({'sst': 'index'})

            data_tsigma_s =\
                xr.open_dataset(data_dir + 'tref_' + s.lower() + '.nc')
            case_events = SelectVariables(aux_cases, data_tsigma_s)

            case_events.to_netcdf(out_dir + 'tref_' + c + '_' + s + '_05.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsTref, [c for c in cases])

else:
    print('one season')
    def SelectEventsTref(c):
        s = seasons[0]
        try:
            aux_cases =\
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                    .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = \
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                    .rename({'sst': 'index'})

        data_tref_s = \
            xr.open_dataset(data_dir + 'tref_' + s.lower() + '_no_detrend.nc')

        case_events = SelectVariables(aux_cases, data_tref_s)
        case_events.to_netcdf(out_dir + 'tref_' + c + '_' + s +
                              '_no_detrend_05.nc')

        data_tref_s = \
            xr.open_dataset(data_dir + 'tref_' + s.lower() + '_detrend.nc')

        case_events = SelectVariables(aux_cases, data_tref_s)
        case_events.to_netcdf(out_dir + 'tref_' + c + '_' + s +
                              '_detrend_05.nc')

    processes = [Process(target=SelectEventsTref, args=(c,)) for c in cases]
    for process in processes:
        process.start()

# PP ###########################################################################
if len(seasons)>1:
    def SelectEventsPrec(c):
        for s in seasons:
            try:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc') \
                    .rename({'__xarray_dataarray_variable__': 'index'})
            except:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc') \
                    .rename({'prec': 'index'})

            data_prec_s =\
                xr.open_dataset(data_dir + 'prec_' + s.lower() + '.nc')
            case_events = SelectVariables(aux_cases, data_prec_s)

            case_events.to_netcdf(out_dir + 'prec_' + c + '_' + s + '_05.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsPrec, [c for c in cases])

else:
    print('one season')
    def SelectEventsPrec(c):
        s = seasons[0]
        try:
            aux_cases =\
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                    .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = \
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                    .rename({'sst': 'index'})

        data_prec_s = \
            xr.open_dataset(data_dir + 'prec_' + s.lower() + '_no_detrend.nc')

        case_events = SelectVariables(aux_cases, data_prec_s)
        case_events.to_netcdf(out_dir + 'prec_' + c + '_' + s +
                              '_no_detrend_05.nc')

        data_prec_s = \
            xr.open_dataset(data_dir + 'prec_' + s.lower() + '_detrend.nc')

        case_events = SelectVariables(aux_cases, data_prec_s)
        case_events.to_netcdf(out_dir + 'prec_' + c + '_' + s +
                              '_detrend_05.nc')

    processes = [Process(target=SelectEventsPrec, args=(c,)) for c in cases]
    for process in processes:
        process.start()

# INDICES --> SCATTER -------------------------------------------------------- #
cases = ['dmi_puros_pos', 'dmi_puros_neg', 'n34_puros_pos', 'n34_puros_neg',
         'sim_pos', 'sim_neg',
         'neutros', 'dmi_neg_n34_pos', 'dmi_pos_n34_neg','dmi_pos', 'dmi_neg',
         'n34_pos', 'n34_neg']

cases_data_dir_dmi = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'

# DMI ##########################################################################
if len(seasons)>1:
    pass
else:
    print('one season')
    def SelectEventsINDEX(c):
        s=seasons[0]
        try:
            aux_cases = xr.open_dataset(
                cases_date_dir + c + '_f_' + s + '_05.nc') \
                .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = xr.open_dataset(
                cases_date_dir + c + '_f_' + s + '_05.nc') \
                .rename({'sst': 'index'})

        #data_sst_s = xr.open_dataset(data_dir + 'sst_' + s.lower() + '.nc')
        data_dmi_s = xr.open_dataset(
            cases_data_dir_dmi + 'DMI_' + s + '_Leads_r_CFSv2.nc')
        case_events = SelectVariables(aux_cases, data_dmi_s)
        case_events.to_netcdf(out_dir + 'dmi_values_' + c + '_' + s + '_05.nc')

    processes = [Process(target=SelectEventsINDEX, args=(c,)) for c in cases]
    for process in processes:
        process.start()

# n34 ##########################################################################
if len(seasons)>1:
    pass
else:
    print('one season')
    def SelectEventsINDEX(c):
        s=seasons[0]
        try:
            aux_cases = xr.open_dataset(
                cases_date_dir + c + '_f_' + s + '_05.nc') \
                .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = xr.open_dataset(
                cases_date_dir + c + '_f_' + s + '_05.nc') \
                .rename({'sst': 'index'})

        #data_sst_s = xr.open_dataset(data_dir + 'sst_' + s.lower() + '.nc')
        data_n34_s = xr.open_dataset(
            cases_data_dir_dmi + 'N34_' + s + '_Leads_r_CFSv2.nc')
        case_events = SelectVariables(aux_cases, data_n34_s)
        #case_events.to_netcdf(out_dir + 'N34_values_' + c + '_' + s + '_05.nc')

    processes = [Process(target=SelectEventsINDEX, args=(c,)) for c in cases]
    for process in processes:
        process.start()


# HGT750 #######################################################################
if len(seasons)>1:
    def SelectEventsHGT750(c):
        for s in seasons:
            try:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'__xarray_dataarray_variable__': 'index'})
            except:
                aux_cases = \
                    xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                        .rename({'sst': 'index'})

            data_hgt_s = xr.open_dataset(data_dir + 'hgt_' + s.lower() + '.nc')
            case_events = SelectVariables(aux_cases, data_hgt_s)

            case_events.to_netcdf(out_dir + 'hgt_' + c + '_' + s + '_05.nc')

    pool = ThreadPool(4)  # uno por season
    pool.map_async(SelectEventsHGT, [c for c in cases])

else:
    print('one season')
    def SelectEventsHGT750(c):
        s = seasons[0]
        try:
            aux_cases = \
                xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                    .rename({'__xarray_dataarray_variable__': 'index'})
        except:
            aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')\
                .rename({'sst': 'index'})

        data_hgt_s = xr.open_dataset(data_dir + 'hgt750_' + s.lower() + '_detrend.nc')

        case_events = SelectVariables(aux_cases, data_hgt_s)
        case_events.to_netcdf(out_dir + 'hgt750_' + c + '_' + s + '__detrend_05.nc')


    processes = [Process(target=SelectEventsHGT750, args=(c,)) for c in cases]
    for process in processes:
        process.start()

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')

# En caso de usar multiples seasons...
# def SelectEventsDMI(s):
#     for c in cases:
#         try:
#             aux_cases = xr.open_dataset(
#                 cases_date_dir + c + '_f_' + s + '_05.nc') \
#                 .rename({'__xarray_dataarray_variable__': 'index'})
#         except:
#             aux_cases = xr.open_dataset(
#                 cases_date_dir + c + '_f_' + s + '_05.nc') \
#                 .rename({'sst': 'index'})
#
#         data_dmi_s = xr.open_dataset(
#             cases_data_dir_dmi + 'DMI_' + s + '_Leads_r_CFSv2.nc')
#
#         case_events = SelectVariables(aux_cases, data_dmi_s)
#
#         case_events.to_netcdf(out_dir + 'dmi_values_' + c + '_' + s + '_05.nc')
#
# # Multiprocess #########################################################################################################
# pool = ThreadPool(4) # uno por season
# pool.map(SelectEventsDMI, [s for s in seasons])
# ########################################################################################################################
#
# # N34 ##################################################################################################################
# cases_data_dir_dmi = '/datos/luciano.andrian/ncfiles/NMME_CFSv2/DMI_N34_Leads_r/'
# def SelectEventsN34(s):
#     for c in cases:
#         try:
#             aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '.nc') \
#                 .rename({'__xarray_dataarray_variable__': 'index'})
#         except:
#             aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '.nc') \
#                 .rename({'sst': 'index'})
#
#         data_n34_s = xr.open_dataset(cases_data_dir_dmi + 'N34_' + s + '_Leads_r_CFSv2.nc')
#
#         case_events = SelectVariables(aux_cases, data_n34_s)
#
#         case_events.to_netcdf(out_dir + 'N34_values_' + c + '_' + s + '.nc')
#
# # Multiprocess #########################################################################################################
# pool = ThreadPool(4) # uno por season
# pool.map(SelectEventsN34, [s for s in seasons])
# ########################################################################################################################
