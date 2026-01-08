"""
Correlacion entre N34 y tref en CFSv2 y OBS
"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'

dates_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
fields_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
import xarray as xr
from matplotlib import colors

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from funciones.indices_utils import DMI, Nino34CPC
from funciones.general_utils import SameDateAs, init_logger
from funciones.plot_utils import PlotFinal, SetDataToPlotFinal
# ---------------------------------------------------------------------------- #
logger = init_logger('12.aux_correlation_temp.log')

# ---------------------------------------------------------------------------- #
if save:
    dpi = 300
else:
    dpi = 100

# ---------------------------------------------------------------------------- #
pastel_div_13 = colors.ListedColormap([
    '#00B28A', '#00C89A', '#00DCAA', '#1AE1AA', '#41E8B7', '#72EEC9',
    'white',
    '#FDD8EB', '#FBB1D7', '#F185BB', '#DF589D', '#CC2C7E', '#B8006E'
])

pastel_div_13.set_under('#00755D')
pastel_div_13.set_over('#7A004C')
pastel_div_13.set_bad('white')

mintrose_divergent_13 = colors.ListedColormap([
    '#00695C', '#00A98A', '#2ED9B1', '#6EEFD0', '#A2F6E0', '#D2FBF1',
    'white',
    '#F9D7EA', '#F5A8CC', '#EC75B0', '#D94293', '#B41576', '#790A56'
])

mintrose_divergent_13.set_under('#004D43')
mintrose_divergent_13.set_over('#4C0035')
mintrose_divergent_13.set_bad('white')

# Funciones ------------------------------------------------------------------ #
def OpenObsDataSet(name, sa=True, dir='/pikachu/datos/luciano.andrian/'
                                      'observado/ncfiles/data_obs_d_w_c/'):

    aux = xr.open_dataset(dir + name + '.nc')

    if sa:
        aux2 = aux.sel(lon=slice(270, 330), lat=slice(15, -60))
        if len(aux2.lat) > 0:
            return aux2
        else:
            aux2 = aux.sel(lon=slice(270, 330), lat=slice(-60, 15))
            return aux2
    else:
        return aux

# Correlation - CFSv2 -------------------------------------------------------- #
logger.info('Correlation CFSv2')
corr_cfsv2 = []
for i_name in ['N34', 'DMI']:
    indice = xr.open_dataset(f'{index_dir}{i_name}_SON_Leads_r_CFSv2.nc')
    indice = (indice - indice.mean())/indice.std()

    for v in ['tref']:

        var = xr.open_dataset(f'{fields_dir}{v}_son_detrend.nc')
        var = (var - var.mean(['r', 'time'])) / var.std(['r', 'time'])

        corr_cfsv2.append(xr.corr(indice.sst, var[v], dim=['time', 'r']))

# Correlation - obs ---------------------------------------------------------- #
logger.info('Correlation obs')
dmi = DMI(filter_bwa=False, start_per=1920, end_per=2020)[2]
dmi = dmi.sel(time=dmi.time.dt.month.isin(10)) # SON
dmi = dmi.sel(time=dmi.time.dt.year.isin(np.arange(1940,2021)))
dmi = (dmi - dmi.mean())/dmi.std()

aux = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
n34 = Nino34CPC(aux, start=1920, end=2020)[0]
n34 = n34.sel(time=n34.time.dt.month.isin(10)) # SON
n34 = (n34 - n34.mean())/n34.std()
n34 = SameDateAs(n34, dmi)

variables_tpp = ['tcru_w_c_d_0.25']
corr_obs = []
for i, i_name in zip([n34, dmi], ['N34', 'DMI']):
    logger.info(f'indice: {i_name}')

    for v in variables_tpp:
        logger.info(f'Variable: {v}')
        data = OpenObsDataSet(name=v + '_SON', sa=False)
        if v == 'tcru_w_c_d_0.25':
            data['time'] = dmi.time

        data = SameDateAs(data, dmi)
        var_name = list(data.data_vars)[0]

        var = data.sel(lon=slice(275, 330), lat=slice(-60, 20))

        if len(var.lat) == 0:
            var = data.sel(lon=slice(275, 330), lat=slice(20, -60))

        var = (var - var.mean(['time'])) / var.std(['time'], skipna=True)

        corr_obs.append(xr.corr(i, var[var_name], dim=['time']))

# ---------------------------------------------------------------------------- #
logger.info('Plot')
aux_v = SetDataToPlotFinal(corr_cfsv2[0], corr_obs[0],
                           corr_cfsv2[1], corr_obs[1])
corr_scale = [-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]
PlotFinal(data=aux_v, levels=corr_scale, cmap=mintrose_divergent_13,
          titles=['','','',''],
          namefig=f'figureS15',
          map='sa', save=save, dpi=dpi, out_dir=out_dir,
          data_ctn=aux_v, color_ctn='k', levels_ctn=corr_scale,
          high=3, width=4, num_cols=2, pdf=True,
          ocean_mask=True, pcolormesh=False)

# ---------------------------------------------------------------------------- #
logger.info('Done')

# ---------------------------------------------------------------------------- #