"""
Correlacion entre N34 y prec y tref en CFSv2
"""
# ---------------------------------------------------------------------------- #
save = False
use_spearman = False
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'

dates_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
fields_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
from matplotlib import colors
from scipy.stats import spearmanr
import numpy as np

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from Funciones import PlotFinal, SetDataToPlotFinal

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
def spearman_correlation(da_field, da_series):

    def spearman_func(x, y):
        # Si hay valores NaN, se ignoran ambos pares
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 3: # se necesitan al menos 3 pares para  Spearman
            return np.nan
        return spearmanr(x[mask], y[mask])[0]

    result = xr.apply_ufunc(
        spearman_func,
        da_field,
        da_series,
        input_core_dims=[["sample"], ["sample"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return result

# Correlation ---------------------------------------------------------------- #

for i_name in ['N34', 'DMI']:
    indice = xr.open_dataset(f'{index_dir}{i_name}_SON_Leads_r_CFSv2.nc')
    indice = (indice - indice.mean())/indice.std()

    corr = []
    for v in ['prec', 'tref']:

        var = xr.open_dataset(f'{fields_dir}{v}_son_detrend.nc')
        var = (var - var.mean(['r', 'time'])) / var.std(['r', 'time'])

        if v == 'tref':
            var = var.sel(time=var.time.dt.year != 2011)
            indice = indice.sel(time=indice.time.dt.year != 2011)

        if use_spearman:
            var_stacked = var[v].stack(sample=('r', 'time'))
            indice_stacked = indice['sst'].stack(sample=('r', 'time'))

            corr.append(spearman_correlation(
                var_stacked.transpose('sample', 'lat', 'lon'),
                indice_stacked.transpose('sample')))
        else:
            corr.append(xr.corr(indice.sst, var[v], dim=['time', 'r']))

    # Plot ------------------------------------------------------------------- #
    aux_v = SetDataToPlotFinal(corr[0], corr[1])
    corr_scale = [-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]
    PlotFinal(data=aux_v, levels=corr_scale,
              cmap=mintrose_divergent_13,
              titles=[f'corr. {i_name} vs Prec', f'corrs {i_name} vs Tref'],
              namefig=f'corr_{i_name}_variables_CFSv2_pcolor',
              map='sa', save=save, dpi=dpi,
              out_dir=out_dir,
              data_ctn=aux_v, color_ctn='k',
              levels_ctn=corr_scale,
              high=4, width=6, num_cols=2, pdf=True,
              ocean_mask=True, pcolormesh=True)

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')