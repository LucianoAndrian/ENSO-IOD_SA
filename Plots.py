"""
Figuras ENSO-IOD-SA
"""
# ---------------------------------------------------------------------------- #
save = False
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'
# ---------------------------------------------------------------------------- #
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import matplotlib.path as mpath

import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")

from Funciones import SetDataToPlotFinal, PlotFinal, CaseComp, RenameDataset

# --
if save:
    dpi = 300
else:
    dpi = 100

# ---------------------------------------------------------------------------- #
def MakerMaskSig(data, r_crit):
    mask_sig = data.where((data < -1 * r_crit) | (data > r_crit))
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

# ---------------------------------------------------------------------------- #
data_dir_proc = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/'

# Scales and colorbars ------------------------------------------------------- #
# Regresion ------------------------------------------------------------------ #
scale_hgt = [-300, -200, -100, -50, -25, -15, 0, 15, 25, 50, 100, 200, 300]
scale_pp = np.linspace(-15, 15, 13)
scale_t = [-.6,-.4,-.2,-.1,-.05,0,0.05,0.1,0.2,0.4,0.6]

# Composite ------------------------------------------------------------------ #
scale_hgt_comp = [-500, -300, -200, -100, -50, 0, 50, 100, 200, 300, 500]
scale_t_comp = [-1.5, -1, -.5, -.25, -.1, 0, .1, .25, .5, 1, 1.5]
scale_pp_comp = [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40]
scale_hgt_comp_cfsv2 = [-375, -275, -175, -75, -25, 0, 25, 75, 175, 275, 375]

# CFSv2
scale_hgt_snr = [-1, -.8, -.6, -.5, -.1, 0, 0.1, 0.5, 0.6, 0.8, 1]
# iods ind
scale_hgt_ind = [-575,-475, -375, -275, -175, -75,0,
                 75, 175, 275, 375, 475, 575]

# ---------------------------------------------------------------------------- #
cbar_sst = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55', '#F28C89',
                                  '#FFCECC', 'white', '#B3DBFF', '#83B9EB',
                                  '#5E9AD7', '#3C7DC3', '#2064AF'][::-1])
cbar_sst.set_over('#9B1C00')
cbar_sst.set_under('#014A9B')
cbar_sst.set_bad(color='white')

cbar_sst = colors.ListedColormap(['#B98200', '#CD9E46', '#E2B361', '#E2BD5A',
                                  '#FFF1C6', 'white', '#B1FFD0', '#7CEB9F',
                                  '#52D770', '#32C355', '#1EAF3D'][::-1])
cbar_sst.set_over('#9B6500')
cbar_sst.set_under('#009B2E')
cbar_sst.set_bad(color='white')


cbar = colors.ListedColormap(['#9B1C00', '#B9391B', '#CD4838', '#E25E55',
                              '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7', '#3C7DC3',
                              '#2064AF', '#014A9B'][::-1])
cbar.set_over('#641B00')
cbar.set_under('#012A52')
cbar.set_bad(color='white')

cbar_snr = colors.ListedColormap(['#070B4F','#2E07AC', '#387AE4' ,'#6FFE9B',
                                  '#FFFFFF',
                                  '#FFFFFF', '#FFFFFF',
                                  '#FEB77E','#CA3E72','#782281','#251255'])
cbar_snr.set_over('#251255')
cbar_snr.set_under('#070B4F')
cbar_snr.set_bad(color='white')

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169', '#79C8BC',
                                 '#B4E2DB',
                                 'white',
                                '#F1DFB3', '#DCBC75', '#995D13', '#6A3D07',
                                 '#543005', ][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

cbar_ks = colors.ListedColormap(['#C2FAB6', '#FAC78F'])

cbar_ks.set_under('#5DCCBF')
cbar_ks.set_over('#FA987C')
cbar_ks.set_bad(color='white')

# Variables comunes ---------------------------------------------------------- #
#sig = True
periodos = [[1940, 2020]]
y1 = 0
p = periodos[0]
s = 'SON'
s_count = 0

# scatter
in_label_size = 18
label_legend_size = 18
tick_label_size = 15
scatter_size_fix = 3

# Titles --------------------------------------------------------------------- #
subtitulos_regre = [r"$ONI$",  r"$DMI$",
                    r"$ONI|_{DMI}$",
                    r"$DMI|_{ONI}$"]

# En orden de ploteo
cases = ['N34_un_pos', 'N34_un_neg', 'DMI_un_pos', 'DMI_un_neg',
         'DMI_sim_pos', 'DMI_sim_neg']

cases_cfsv2 =['n34_puros_pos', 'n34_puros_neg',
              'dmi_puros_pos', 'dmi_puros_neg',
              'sim_pos', 'sim_neg']

title_case = ['Pure El Niño', 'Pure La Niña',
              'Pure positive IOD ', 'Pure negative IOD',
              'El Niño - positive IOD', 'La Niña - negative IOD']

# ---------------------------------------------------------------------------- #
# Regresion ------------------------------------------------------------------ #
t_critic = 1.66  # es MUY similar (2 digitos) para ambos períodos
r_crit = np.sqrt(1 / (((np.sqrt((p[1] - p[0]) - 2) / t_critic) ** 2) + 1))

variables = ['prec', 'temp', 'hgt200', 'hgt750']
aux_scales = [scale_pp, scale_t, scale_hgt, scale_hgt]
aux_cbar = [cbar_pp, cbar, cbar, cbar]
for v, v_scale, v_cbar in zip(variables, aux_scales, aux_cbar):


    regre_n34 = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_n34.nc')
    regre_corr_n34 = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_n34_corr.nc')
    regre_dmi = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_dmi.nc')
    regre_corr_dmi = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_dmi_corr.nc')
    regre_n34_wodmi = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_n34_wodmi.nc')
    regre_corr_n34_wodmi = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_n34_wodmi_corr.nc')
    regre_dmi_won34 = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_dmi_won34.nc')
    regre_corr_dmi_won34 = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_dmi_won34_corr.nc')

    regre_n34, regre_corr_n34, regre_dmi, regre_corr_dmi, regre_n34_wodmi, \
        regre_corr_n34_wodmi, regre_dmi_won34, regre_corr_dmi_won34 = \
        RenameDataset('var', regre_n34, regre_corr_n34, regre_dmi,
                      regre_corr_dmi, regre_n34_wodmi, regre_corr_n34_wodmi,
                      regre_dmi_won34, regre_corr_dmi_won34)

    aux0 = regre_n34 * MakerMaskSig(regre_corr_n34, r_crit)
    aux1 = regre_dmi * MakerMaskSig(regre_corr_dmi, r_crit)
    aux2 = regre_n34_wodmi * MakerMaskSig(regre_corr_n34_wodmi, r_crit)
    aux3 = regre_dmi_won34 * MakerMaskSig(regre_corr_dmi_won34, r_crit)

    aux_v = SetDataToPlotFinal(aux0, aux1, aux2, aux3)

    PlotFinal(data=aux_v, levels=v_scale, cmap=v_cbar,
              titles=subtitulos_regre, namefig='f02', map='sa',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=aux_v, levels_ctn=v_scale, color_ctn='k',
              data_ctn2=None, levels_ctn2=None,
              color_ctn2=['#FF0002', '#0003FF'], high=5)

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

