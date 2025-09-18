"""
Correlacion entre N34, dmi y prec y tref
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'

variables_tpp = ['ppgpcc_w_c_d_1', 'tcru_w_c_d_0.25']
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
from matplotlib import colors
import numpy as np

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from Funciones import PlotFinal, SetDataToPlotFinal, Nino34CPC, DMI, SameDateAs
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

# Correlation ---------------------------------------------------------------- #
print(' DMI y N34 ---------------------------------------------------------- #')
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


for i, i_name in zip([n34, dmi], ['N34', 'DMI']):

    corr = []
    for v in variables_tpp:
        data = OpenObsDataSet(name=v + '_SON', sa=False)
        if v == variables_tpp[1]:
            data['time'] = dmi.time

        data = SameDateAs(data, dmi)
        var_name = list(data.data_vars)[0]

        var = data.sel(lon=slice(275, 330), lat=slice(-60, 20))

        if len(var.lat) == 0:
            var = data.sel(lon=slice(275, 330), lat=slice(20, -60))

        var = (var - var.mean(['time'])) / var.std(['time'], skipna=True)

        corr.append(xr.corr(i, var[var_name], dim=['time']))

    # Plot ------------------------------------------------------------------- #
    aux_v = SetDataToPlotFinal(corr[0], corr[1])
    corr_scale = [-1, -0.75, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 0.75, 1]
    PlotFinal(data=aux_v, levels=corr_scale,
              cmap=mintrose_divergent_13,
              titles=[f'corr. {i_name} vs Prec', f'corrs {i_name} vs Tref'],
              namefig=f'corr_{i_name}_variables_OBS_pcolor',
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