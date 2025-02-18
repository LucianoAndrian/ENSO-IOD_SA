"""
Figuras ENSO-IOD-SA
"""
from fontTools.varLib.varStore import VarStore_subset_varidxes

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

def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

# ---------------------------------------------------------------------------- #
data_dir_proc = '/pikachu/datos/luciano.andrian/paper2/salidas_nc/'

nc_date_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
              'nc_composites_dates_no_ind_sst_anom/'

sig_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/nc_quantiles/' \
          'DMIbase/' # resultados de MC

data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1940_2020/'

cases_dir = "/pikachu/datos/luciano.andrian/cases_fields/"
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
# Scales and colorbars ------------------------------------------------------- #
# Regresion ------------------------------------------------------------------ #
scale_hgt = [-300, -200, -100, -50, -25, 0, 25, 50, 100, 200, 300]
scale_hgt_750 = [ -100, -75, -50, -25, -10, 0, 10, 25, 50, 75, 100 ]
scale_pp = np.linspace(-15, 15, 13)
scale_t = [-.6,-.4,-.2,-.1,-.05,0,0.05,0.1,0.2,0.4,0.6]

scale_pp_val = [-60, -30, -10, -5, 0, 5, 10, 30, 60]
scale_t_val = [-2,-.8,-.4,-.1, 0, .1, .4, .8, 2]
# Composite ------------------------------------------------------------------ #
scale_hgt_comp = [-500, -300, -200, -100, -50, 0, 50, 100, 200, 300, 500]
scale_t_comp = [-1.5, -1, -.5, -.25, -.1, 0, .1, .25, .5, 1, 1.5]
scale_pp_comp = [-40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40]
scale_hgt_comp_cfsv2 = [-375, -275, -175, -75, -25, 0, 25, 75, 175, 275, 375]

# CFSv2
scale_snr = [-1, -.8, -.6, -.5, -.1, 0, 0.1, 0.5, 0.6, 0.8, 1]
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


cbar_snr_t = colors.ListedColormap(['#00876c', '#439981', '#6aaa96', '#8cbcac',
                                    '#aecdc2', '#cfdfd9', '#FFFFFF',
                                    '#f1d4d4', '#f0b8b8', '#ec9c9d',
                                    '#e67f83', '#de6069', '#d43d51'])

cbar_snr_t.set_under('#006D53')
cbar_snr_t.set_over('#AB183F')
cbar_snr_t.set_bad(color='white')


cbar_snr_pp = colors.ListedColormap(['#a97300', '#bb8938', '#cc9f5f',
                                     '#dbb686','#e9cead',
                                     '#f5e6d6', '#ffffff', '#dce9eb',
                                     '#b8d4d8', '#95bfc5','#70aab2',
                                     '#48959f', '#00818d'])

cbar_snr_pp.set_under('#6A3D07')
cbar_snr_pp.set_over('#1E6D5A')
cbar_snr_pp.set_bad(color='white')

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


cases_cfsv2 =['dmi_puros_pos', 'n34_puros_pos', 'sim_pos',
              'dmi_puros_neg', 'n34_puros_neg', 'sim_neg']

title_case = ['Pure positive IOD ', 'Pure El Niño', 'El Niño - positive IOD',
               'Pure negative IOD', 'Pure La Niña', 'La Niña - negative IOD']

# ---------------------------------------------------------------------------- #

print('# Validation -------------------------------------------------------- #')
variables = ['prec', 'tref']
aux_scales = [scale_pp_val, scale_t_val]
aux_cbar = [cbar_pp, cbar, cbar, cbar]
for v, v_scale, v_cbar in zip(variables, aux_scales, aux_cbar):
    print('Climatology ------------------------------------------------------ ')
    clim_dif = xr.open_dataset(f'{data_dir_proc}{v}_dif_clim_no-norm.nc')
    clim_dif_test = xr.open_dataset(
        f'{data_dir_proc}{v}_pvalue_clim_no-norm.nc')

    print('Events ----------------------------------------------------------- ')
    # CFSv2
    enso_cfsv2_dif_test =  xr.open_dataset(
        f'{data_dir_proc}{v}_enso_cfsv2_test.nc')
    enso_cfsv2_dif = xr.open_dataset(f'{data_dir_proc}{v}_enso_cfsv2_dif.nc')

    iod_cfsv2_dif_test = xr.open_dataset(f'{data_dir_proc}{v}_iod_cfsv2_test.nc')
    iod_cfsv2_dif = xr.open_dataset(f'{data_dir_proc}{v}_iod_cfsv2_dif.nc')

    # Obs
    enso_obs_dif_test = xr.open_dataset(f'{data_dir_proc}{v}_enso_obs_test.nc')
    enso_obs_dif = xr.open_dataset(f'{data_dir_proc}{v}_enso_obs_dif.nc')

    iod_obs_dif_test = xr.open_dataset(f'{data_dir_proc}{v}_iod_obs_test.nc')
    iod_obs_dif = xr.open_dataset(f'{data_dir_proc}{v}_iod_obs_dif.nc')

    clim_dif, clim_dif_test, enso_cfsv2_dif_test, enso_cfsv2_dif, \
        iod_cfsv2_dif_test, iod_cfsv2_dif, enso_obs_dif_test, enso_obs_dif, \
        iod_obs_dif_test, iod_obs_dif = \
        RenameDataset('var', clim_dif, clim_dif_test, enso_cfsv2_dif_test,
                      enso_cfsv2_dif, iod_cfsv2_dif_test, iod_cfsv2_dif,
                      enso_obs_dif_test, enso_obs_dif, iod_obs_dif_test,
                      iod_obs_dif)

    aux_sig = SetDataToPlotFinal(clim_dif.where(clim_dif_test['var'].values<0.1),
        enso_obs_dif.where(enso_obs_dif_test['var'].values < 0.1),
        iod_obs_dif.where(iod_obs_dif_test['var'].values < 0.1),
        iod_cfsv2_dif * np.nan,
        enso_cfsv2_dif.where(enso_cfsv2_dif_test['var'].values < 0.1),
        iod_cfsv2_dif.where(iod_cfsv2_dif_test['var'].values < 0.1))

    aux_v = SetDataToPlotFinal(clim_dif, enso_obs_dif, iod_obs_dif,
                               iod_cfsv2_dif*0,
                               enso_cfsv2_dif, iod_cfsv2_dif)

    PlotFinal(data=aux_v, levels=v_scale, cmap=v_cbar,
              titles=['', '', '', '', '', ''], namefig=f'val_{v}', map='sa',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=None, levels_ctn=v_scale, color_ctn='k',
              data_ctn2=None, levels_ctn2=None,
              color_ctn2=None, high=3, width=7, num_cols=3, pdf=True,
              sig_points=aux_sig, hatches='...', ocean_mask=True)

print('Done Validation ------------------------------------------------------ ')
print(' --------------------------------------------------------------------- ')
print('                                                                       ')
# ---------------------------------------------------------------------------- #

print('# Regresion --------------------------------------------------------- #')
t_critic = 1.66  # es MUY similar (2 digitos) para ambos períodos
r_crit = np.sqrt(1 / (((np.sqrt((p[1] - p[0]) - 2) / t_critic) ** 2) + 1))

variables = ['prec', 'temp', 'hgt200', 'hgt750']
aux_scales = [scale_pp, scale_t, scale_hgt, scale_hgt_750]
aux_cbar = [cbar_pp, cbar, cbar, cbar]
for v, v_scale, v_cbar in zip(variables, aux_scales, aux_cbar):
    print(f'{v} ------------------------------------------------------------- ')
    regre_n34 = xr.open_dataset(f'{data_dir_proc}regre/{v}_regre_n34.nc')
    regre_corr_n34 = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_n34_corr.nc')
    regre_dmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi.nc')
    regre_corr_dmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi_corr.nc')
    regre_n34_wodmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_n34_wodmi.nc')
    regre_corr_n34_wodmi = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_n34_wodmi_corr.nc')
    regre_dmi_won34 = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi_won34.nc')
    regre_corr_dmi_won34 = xr.open_dataset(
        f'{data_dir_proc}regre/{v}_regre_dmi_won34_corr.nc')

    regre_n34, regre_corr_n34, regre_dmi, regre_corr_dmi, regre_n34_wodmi, \
        regre_corr_n34_wodmi, regre_dmi_won34, regre_corr_dmi_won34 = \
        RenameDataset('var', regre_n34, regre_corr_n34, regre_dmi,
                      regre_corr_dmi, regre_n34_wodmi, regre_corr_n34_wodmi,
                      regre_dmi_won34, regre_corr_dmi_won34)

    aux_sig = SetDataToPlotFinal(
        regre_n34 * MakerMaskSig(regre_corr_n34, r_crit),
        regre_dmi * MakerMaskSig(regre_corr_dmi, r_crit),
        regre_n34_wodmi * MakerMaskSig(regre_corr_n34_wodmi, r_crit),
        regre_dmi_won34 * MakerMaskSig(regre_corr_dmi_won34, r_crit))

    aux_v = SetDataToPlotFinal(regre_n34, regre_dmi, regre_n34_wodmi,
                               regre_dmi_won34)

    PlotFinal(data=aux_v, levels=v_scale, cmap=v_cbar,
              titles=subtitulos_regre, namefig=f'regre_{v}', map='sa',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=aux_v, levels_ctn=v_scale, color_ctn='k',
              data_ctn2=None, levels_ctn2=None,
              color_ctn2=None, high=3.1, width = 4,
              sig_points=aux_sig, hatches='...', pdf=True)

print('Done Regression ------------------------------------------------------ ')
print(' --------------------------------------------------------------------- ')
print('                                                                       ')
# ---------------------------------------------------------------------------- #

print('# Obs. Composite ---------------------------------------------------- #')
variables_tpp = ['tcru_w_c_d_0.25', 'ppgpcc_w_c_d_1']
plt.rcParams['hatch.linewidth'] = 1
# hacer funcion con esto


cases = ['DMI_un_pos', 'N34_un_pos', 'DMI_sim_pos',
         'DMI_un_neg', 'N34_un_neg', 'DMI_sim_neg']

title_case = ['Pure positive IOD', 'Pure El Niño',  'El Niño - pos. IOD',
              'Pure negative IOD', 'Pure La Niña', 'La Niña - neg. IOD']

aux_scales = [scale_t_comp, scale_pp_comp]
aux_cbar = [cbar, cbar_pp]

variables_hgt = ['HGT200', 'HGT750']
aux_scales_hgt = [scale_hgt, scale_hgt_750]
aux_cbar_hgt = [cbar, cbar]
for v_count, (v, v_scale, v_cbar) in enumerate(
        zip(variables_tpp,aux_scales, aux_cbar)):

    data = OpenObsDataSet(name=v + '_SON', sa=False)

    # HGT 750
    for v_hgt, v_hgt_scale, v_hgt_cbar in zip(
            variables_hgt, aux_scales_hgt, aux_cbar_hgt):

        aux_var = []
        aux_var_no_sig = []
        aux_var_sig = []
        aux_hgt_no_sig = []
        aux_hgt_sig = []

        data_hgt = xr.open_dataset(f'{data_dir}{v_hgt}_SON_mer_d_w.nc')
        data_hgt = data_hgt.sel(lon=slice(275, 330), lat=slice(20, -65))
        for c_count, c in enumerate(cases):
            comp1, num_case = CaseComp(data, s, mmonth=[9, 11], c=c,
                                       nc_date_dir=nc_date_dir)

            data_sig = xr.open_dataset(sig_dir + v.split('_')[0] + '_' + c +
                                       '1940_2020_SON.nc')

            comp1_i = comp1.interp(lon=data_sig.lon.values,
                                   lat=data_sig.lat.values)

            # ver esto
            sig = comp1_i.where((comp1_i < data_sig['var'][0]) |
                                (comp1_i > data_sig['var'][1]))
            sig = sig.where(np.isnan(sig['var']), 1)

            aux_var_no_sig.append(comp1)
            aux_var_sig.append(sig)

            # hgt: # se puede usar el two_variables...
            comp1, num_case = CaseComp(data_hgt, s, mmonth=[9, 11], c=c,
                                       nc_date_dir=nc_date_dir)

            data_sig = xr.open_dataset(f'{sig_dir}{v_hgt}_{c}1940_2020_{s}.nc')

            sig = comp1.where((comp1 < data_sig['var'][0]) |
                              (comp1 > data_sig['var'][1]))
            sig = sig.where(np.isnan(sig['var']), 1)

            aux_hgt_no_sig.append(comp1)
            aux_hgt_sig.append(sig)

        aux_var_no_sig = xr.concat(aux_var_no_sig, dim='plots')
        aux_var_sig = xr.concat(aux_var_sig, dim='plots')
        aux_hgt_no_sig = xr.concat(aux_hgt_no_sig, dim='plots')
        aux_hgt_sig = xr.concat(aux_hgt_sig, dim='plots')

        PlotFinal(data=aux_var_no_sig, levels=v_scale, cmap=v_cbar,
                  titles=title_case, namefig=f'val_{v}', map='sa',
                  save=save, dpi=dpi, out_dir=out_dir,
                  data_ctn=None, levels_ctn=v_scale, color_ctn='k',
                  data_ctn2=aux_hgt_no_sig, levels_ctn2=v_hgt_scale,
                  color_ctn2='k', high=3, width=7, num_cols=3, pdf=True,
                  sig_points=aux_var_sig, hatches='...', ocean_mask=True,
                  data_ctn2_no_ocean_mask=True)

        PlotFinal(data=aux_hgt_no_sig, levels=v_hgt_scale, cmap=v_hgt_cbar,
                  titles=title_case, namefig=f'val_{v}', map='sa',
                  save=save, dpi=dpi, out_dir=out_dir,
                  data_ctn=aux_hgt_no_sig, levels_ctn=v_hgt_scale, color_ctn='k',
                  data_ctn2=None, levels_ctn2=None,
                  color_ctn2=None, high=3, width=7, num_cols=3, pdf=True,
                  sig_points=aux_hgt_sig, hatches='...', ocean_mask=False)

print('Done Obs. Composite -------------------------------------------------- ')
print(' --------------------------------------------------------------------- ')
print('                                                                       ')
# ---------------------------------------------------------------------------- #

print('# CFSv2 Composite --------------------------------------------------- #')
variables = ['tref', 'prec']
aux_scales = [scale_t, scale_pp]
aux_cbar = [cbar, cbar_pp]
aux_cbar_snr = [cbar_snr_t, cbar_snr_pp]

variables_hgt = ['hgt'] # vamos a necesitar otro nivel?
aux_scales_hgt = [scale_hgt_comp_cfsv2]
aux_cbar_hgt = [cbar]

for v, v_scale, v_cbar, v_cbar_snr in zip(variables, aux_scales, aux_cbar,
                                          aux_cbar_snr):
    if v == 'prec':
        fix = 30
    else:
        fix = 1

    data_neutro = xr.open_dataset(
        f'{cases_dir}{v}_neutros_SON_detrend_05.nc')*fix
    data_neutro = data_neutro.rename({list(data_neutro.data_vars)[0]: 'var'})

    for v_hgt, v_hgt_scale, v_hgt_cbar in zip(
            variables_hgt, aux_scales_hgt, aux_cbar_hgt):

        neutro_hgt = (xr.open_dataset(f'{cases_dir}{v_hgt}_neutros_SON_05.nc')
                      .rename({'hgt': 'var'}))
        neutro_hgt = Weights(neutro_hgt.__mul__(9.80665))
        neutro_hgt = neutro_hgt.rename({list(neutro_hgt.data_vars)[0]: 'var'})

        aux_data = []
        aux_data_snr = []
        aux_hgt = []
        aux_hgt_snr = []
        aux_num_cases = []

        for c in cases_cfsv2:
            case = xr.open_dataset(f'{cases_dir}{v}_{c}_SON_detrend_05.nc')
            case = case.rename({list(case.data_vars)[0]: 'var'})
            case = case*fix

            case_hgt = xr.open_dataset(f'{cases_dir}{v_hgt}_{c}_SON_05.nc')
            case_hgt = case_hgt.rename({list(case_hgt.data_vars)[0]: 'var'})
            case_hgt = Weights(case_hgt.__mul__(9.80665))

            #case = Weights(case.__mul__(9.80665))
            num_case = len(case.time)
            aux_num_cases.append(num_case)

            comp = case.mean('time') - data_neutro.mean('time')
            comp_hgt = case_hgt.mean('time') - neutro_hgt.mean('time')

            aux_spread = case - comp
            aux_spread = aux_spread.std('time')
            snr = comp / aux_spread

            aux_spread = case_hgt - comp_hgt
            aux_spread = aux_spread.std('time')
            snr_hgt = comp_hgt / aux_spread

            aux_data.append(comp)
            aux_data_snr.append(snr)
            aux_hgt.append(comp_hgt)
            aux_hgt_snr.append(snr_hgt)

        aux_hgt = xr.concat(aux_hgt, dim='plots')
        aux_hgt_snr = xr.concat(aux_hgt_snr, dim='plots')

        aux_data = xr.concat(aux_data, dim='plots')
        aux_data_snr = xr.concat(aux_data_snr, dim='plots')

    aux_scale_hgt=[-150, -100, -50, -25, -10, 10, 25, 50, 100, 150]
    PlotFinal(data=aux_data, levels=v_scale, cmap=v_cbar,
              titles=title_case, namefig=f"f10", map='sa',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=aux_hgt, color_ctn='k',
              high=3, width=7, num_cols=3,
              num_cases=True, num_cases_data=aux_num_cases,
              levels_ctn=aux_scale_hgt, ocean_mask=True,
              data_ctn_no_ocean_mask=True)

    PlotFinal(data=aux_data_snr, levels=scale_snr, cmap=v_cbar_snr,
              titles=title_case, namefig=f"f10", map='sa',
              save=save, dpi=dpi, out_dir=out_dir,
              data_ctn=aux_hgt, color_ctn='k',
              high=3, width=7, num_cols=3,
              num_cases=True, num_cases_data=aux_num_cases,
              levels_ctn=v_hgt_scale, ocean_mask=True,
              data_ctn_no_ocean_mask=True)

    if v == variables[0]:
        # scale_hgt=[-150, -100, -50, -25, -10,
        # 10, 25, 50, 100, 150]
        PlotFinal(data=aux_hgt, levels=scale_hgt, cmap=v_hgt_cbar,
                  titles=title_case, namefig=f"f10", map='sa',
                  save=save, dpi=dpi, out_dir=out_dir,
                  data_ctn=aux_hgt, color_ctn='k',
                  high=3, width=7, num_cols=3,
                  num_cases=True, num_cases_data=aux_num_cases,
                  levels_ctn=scale_hgt, ocean_mask=False)

        PlotFinal(data=aux_hgt_snr, levels=scale_snr, cmap=cbar_snr,
                  titles=title_case, namefig=f"f10", map='sa',
                  save=save, dpi=dpi, out_dir=out_dir,
                  data_ctn=aux_hgt_snr, color_ctn='k',
                  high=3, width=7, num_cols=3,
                  num_cases=True, num_cases_data=aux_num_cases,
                  levels_ctn=scale_snr, ocean_mask=False)

print('Done CFSv2 Composite ------------------------------------------------- ')
print(' --------------------------------------------------------------------- ')