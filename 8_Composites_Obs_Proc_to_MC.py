"""
Composiciones observadas
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = ('/pikachu/datos/luciano.andrian/observado/ncfiles/'
           'nc_composites_dates_no_ind_sst_anom/')
# ---------------------------------------------------------------------------- #
import xarray as xr
import pandas as pd
pd.options.mode.chained_assignment = None
from funciones.indices_utils import Nino34CPC, DMI
from funciones.composite_utils import MultipleComposite
from funciones.general_utils import init_logger

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )

# ---------------------------------------------------------------------------- #
w_dir = '/home/luciano.andrian/doc/salidas/'
pwd = '/datos/luciano.andrian/ncfiles/'
# ---------------------------------------------------------------------------- #
logger = init_logger('8_Composites_Obs_Proc_to_MC.log')

# ---------------------------------------------------------------------------- #
seasons = [10]
seasons_name = ['SON']
full_season2 = False
bwa = False
fs = False

start = 1920 # *no afecta al computo con 1940
end = 2020

logger.info('DMI y N34')
dmi, aux, dmi_aux = DMI(filter_bwa=False, start_per=start, end_per=end)
del aux
aux = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
n34 = Nino34CPC(aux, start=start, end=end)[2]
del aux


logger.info('Composite')
for s_count, s in enumerate(seasons):
    logger.info(f'seasons_name[s_count]')

    Neutral, DMI_sim_pos, DMI_sim_neg, DMI_un_pos, DMI_un_neg, N34_un_pos, \
        N34_un_neg, DMI_pos, DMI_neg, N34_pos, N34_neg, DMI_pos_N34_neg, \
        DMI_neg_N34_pos = MultipleComposite(var=dmi_aux, n34=n34, dmi=dmi,
                                            season=s - 1, # dentro ok
                                            start=start,
                                            full_season=False,
                                            compute_composite=False)

    ds = xr.Dataset(
        data_vars={
            'Neutral': (Neutral),
            "DMI_sim_pos": (DMI_sim_pos),
            "DMI_sim_neg": (DMI_sim_neg),
            "DMI_un_pos": (DMI_un_pos),
            "DMI_un_neg": (DMI_un_neg),
            "N34_un_pos": (N34_un_pos),
            "N34_un_neg": (N34_un_neg),
            "DMI_pos": (DMI_pos),
            "DMI_neg": (DMI_neg),
            "N34_pos": (N34_pos),
            "N34_neg": (N34_neg),
            "DMI_pos_N34_neg": (DMI_pos_N34_neg),
            "DMI_neg_N34_pos": (DMI_neg_N34_pos),
        }
    )

    if save is True:
        logger.info('Saving...')
        ds.to_netcdf(f'{out_dir}{start}_{end}_{seasons_name[s_count]}.nc')

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #