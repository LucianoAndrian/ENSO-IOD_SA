"""
Correlacion entre N34 y prec y tref en CFSv2
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'

dates_dir = '/pikachu/datos/luciano.andrian/cases_dates/'
fields_dir = '/pikachu/datos/luciano.andrian/cases_fields/'
index_dir = '/pikachu/datos/luciano.andrian/DMI_N34_Leads_r/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
from matplotlib import colors


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

# Correlation ---------------------------------------------------------------- #
n34 = xr.open_dataset(f'{index_dir}N34_SON_Leads_r_CFSv2.nc')
n34 = (n34 - n34.mean())/n34.std()
corr = []
for v in ['prec', 'tref']:

    var = xr.open_dataset(f'{fields_dir}{v}_son_detrend.nc')
    var = (var - var.mean(['r', 'time'])) / var.std(['r', 'time'])

    if v == 'tref':
        var = var.sel(time=var.time.dt.year != 2011)
        n34 = n34.sel(time=n34.time.dt.year != 2011)

    corr.append(xr.corr(n34.sst, var[v], dim=['time', 'r']))


# Plot ----------------------------------------------------------------------- #
aux_v = SetDataToPlotFinal(corr[0], corr[1])
corr_scale = [-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]
PlotFinal(data=aux_v, levels=corr_scale,
          cmap=mintrose_divergent_13,
          titles=['corr. N34 vs Prec', 'corrs N34 vs Tref'],
          namefig=f'corr_n34_variables',
          map='sa', save=False, dpi=100,
          out_dir=None,
          data_ctn=aux_v, color_ctn='k',
          levels_ctn=corr_scale,
          high=5, width=8, num_cols=2, pdf=True,
          ocean_mask=True, pcolormesh=False)

print('# --------------------------------------------------------------------#')
print('# --------------------------------------------------------------------#')
print('done')
print('# --------------------------------------------------------------------#')