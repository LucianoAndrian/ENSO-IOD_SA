"""
Testeo replica regresion eof, fail
'"""
"""
Antes de usar eof, total es casi lo mismo.
¿los composites sinteticos luego de la regresion pueden contener años que en
 la realidad son opuestos?
"""
import statsmodels.formula.api as smf
import pandas as pd
from eofs.xarray import Eof
import xarray as xr
from Funciones import DMI, Nino34CPC
import numpy as np
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib import colors

# ---------------------------------------------------------------------------- #
def plotindex(index, indexwo, title='title', label='label'):
    plt.plot(index, color='k',
             label='original')
    plt.plot(indexwo, color='r',
             label=label)
    plt.grid()
    plt.ylim((-3,3))
    plt.legend()
    plt.title(title)
    plt.show()

def LinearReg(xrda, dim, deg=1):
    # liner reg along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg, skipna=True)
    return aux

def LinearReg1_D(dmi, n34):

    df = pd.DataFrame({'dmi': dmi.values, 'n34': n34.values})

    result = smf.ols(formula='n34~dmi', data=df).fit()
    n34_pred_dmi = result.params[1] * dmi.values + result.params[0]

    result = smf.ols(formula='dmi~n34', data=df).fit()
    dmi_pred_n34 = result.params[1] * n34.values + result.params[0]

    return n34 - n34_pred_dmi, dmi - dmi_pred_n34

def RegWOEffect(n34, n34_wo_dmi, dmi, dmi_wo_n34, m=9, datos=None):

    datos['time'] = n34

    try:
        #aux = LinearReg(datos.groupby('month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) +\
        #       aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        #aux = LinearReg(datos.groupby('time.month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('time.month')[m].time, aux.var_polyfit_coefficients[0]) +\
        #       aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time, aux.var_polyfit_coefficients[0]) +\
              aux.var_polyfit_coefficients[1]
    #wo n34
    try:
        #var_regdmi_won34 = datos.groupby('month')[m]-aux
        var_regdmi_won34 = datos - aux

        #var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m] #index wo influence
        var_regdmi_won34['time'] = dmi_wo_n34
        var_dmi_won34 = LinearReg(var_regdmi_won34,'time')
    except:
        #var_regdmi_won34 = datos.groupby('time.month')[m] - aux
        var_regdmi_won34 = datos - aux

        #var_regdmi_won34['time'] = dmi_wo_n34.groupby('time.month')[m]  # index wo influence
        var_regdmi_won34['time'] = dmi_wo_n34  # index wo influence
        var_dmi_won34 = LinearReg(var_regdmi_won34, 'time')

    #-----------------------------------------#

    datos['time'] = dmi
    try:
        #aux = LinearReg(datos.groupby('month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
        #   aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        aux = LinearReg(datos.groupby('time.month')[m], 'time')
        aux = LinearReg(datos, 'time')
        # aux = xr.polyval(datos.groupby('time.month')[m].time, aux.var_polyfit_coefficients[0]) + \
        #   aux.var_polyfit_coefficients[1]
        aux = xr.polyval(datos.time,
                         aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    #wo
    try:
        # var_regn34_wodmi = datos.groupby('month')[m]-aux
        # var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
        var_regn34_wodmi = datos-aux
        var_regn34_wodmi['time'] = n34_wo_dmi #index wo influence
        var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    except:
        # var_regn34_wodmi = datos.groupby('time.month')[m]-aux
        # var_regn34_wodmi['time'] = n34_wo_dmi.groupby('time.month')[m] #index wo influence
        var_regn34_wodmi = datos - aux
        var_regn34_wodmi['time'] = n34_wo_dmi #index wo influence
        var_n34_wodmi = LinearReg(var_regn34_wodmi,'time')

    return var_n34_wodmi.var_polyfit_coefficients[0],\
           var_dmi_won34.var_polyfit_coefficients[0],\
           var_regn34_wodmi,var_regdmi_won34

def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt


def WaveFilter(serie, harmonic):
    N = np.size(serie)

    sum_sin = np.sum(serie * np.sin(harmonic * 2 * np.pi * np.arange(N) / N))
    sum_cos = np.sum(serie * np.cos(harmonic * 2 * np.pi * np.arange(N) / N))

    A = 2 * sum_sin / N
    B = 2 * sum_cos / N

    xs = A * np.sin(2 * np.pi * harmonic * np.arange(N) / N) + \
         B * np.cos(2 * np.pi * harmonic * np.arange(N) / N)

    return serie - xs


def bandpass_filter(serie, max_period):
    min_period = 2
    N = np.size(serie)

    if np.all(np.isnan(serie)) or N < max_period:
        return serie  # sin cambios si está vacío o no hay suficiente data

    # Rango de armónicos a eliminar: fuera del rango 2-7 años
    min_harmonic = int(np.floor(N / max_period))
    max_harmonic = int(np.floor(N / min_period))

    filtered = serie.copy()

    for h in range(1,
                   min_harmonic):  # Remueve componentes de baja frecuencia (>7 años)
        filtered = WaveFilter(filtered, h)

    for h in range(max_harmonic + 1,
                   int(N / 2) + 1):  # Remueve alta frecuencia (<2 años)
        filtered = WaveFilter(filtered, h)

    return filtered

def PlotBars(indice):
    values = -1 * indice / indice.std()
    years = indice.time.dt.year
    colors = ['blue' if v > 0 else 'red' for v in values]
    plt.figure(figsize=(10, 4))
    plt.bar(x=years, height=values, color=colors)
    plt.grid(True)
    plt.ylim((-2, 3))
    plt.xlabel("Año")
    plt.show()

def plot_sst_times(ds, levels=np.arange(-1,1.2,0.2)):
    """
    Plotea los 10 tiempos de la variable 'var' en el Dataset dado, usando contourf.

    Parámetros:
        ds (xarray.Dataset): Dataset con dimensiones (time, lat, lon) y variable 'var'
    """
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15, 6),
                             subplot_kw={'projection':  ccrs.PlateCarree(
                                 central_longitude=180)})
    axes = axes.flatten()

    for i, ax in enumerate(axes):

        try:
            time = ds.time[i].values
            try:
                sst = ds.sel(time=time)#['var']
                cf = ax.contourf(ds.lon, ds.lat, sst,
                                cmap='RdBu_r', levels=levels, extend='both',
                                transform=ccrs.PlateCarree())
            except:
                sst = ds.sel(time=time)['var']
                cf = ax.contourf(ds.lon, ds.lat, sst,
                                 cmap='RdBu_r', levels=levels, extend='both',
                                 transform=ccrs.PlateCarree())



            ax.set_title(str(ds.time[i].dt.strftime('%Y-%m-%d').values))
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linewidth=0.5)
            #ax.set_global()

        except:
            pass

    cbar = fig.colorbar(cf, ax=axes, orientation='horizontal',
                        fraction=0.05,
                        pad=0.05)

    cbar.set_label('SST anomaly (°C)')

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.25, top=0.93,
                        wspace=0.15, hspace=0.25)
    plt.show()

def RegreField(field, index):
    data = field.copy()

    data['time'] = index
    aux = LinearReg(data, 'time')
    try:
        aux = xr.polyval(data.time, aux.var_polyfit_coefficients[0]) + \
              aux.var_polyfit_coefficients[1]
    except:
        aux = xr.polyval(data.time, aux.polyfit_coefficients[0]) + \
              aux.polyfit_coefficients[1]
    aux['time'] = field.time.values
    return aux


def find_top_events(index, num_of_events):
    pos = index.sortby(index, ascending=False).isel(time=slice(num_of_events))
    neg = index.sortby(index, ascending=True).isel(time=slice(num_of_events))

    return pos, neg

def select_to_events(field, index_pos, index_neg):
    try:
        top_pos_events = field.sel(time=index_pos.time)
        top_neg_events = field.sel(time=index_neg.time)
    except:
        top_pos_events = field.sel(time=field.time.dt.year.isin(index_pos))
        top_neg_events = field.sel(time=field.time.dt.year.isin(index_neg))

    return top_pos_events, top_neg_events



def PlotSSTOne(field, levels = np.arange(-1,1.1,0.1), dpi=100):

    fig = plt.figure(figsize=(8, 4), dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    crs_latlon = ccrs.PlateCarree()
    ax.set_extent([0, 359, -50, 50], crs=crs_latlon)

    cbar = [
        # deep → pale blue              |  WHITE  |  pale → deep red
        '#014A9B', '#155AA8', '#276BB4', '#397AC1', '#4E8CCE',
        '#649FDA', '#7BB2E7', '#97C7F3', '#B7DDFF',
        '#FFFFFF',  # −0.1 ↔ 0.0
        '#FFFFFF',  # 0.0 ↔ 0.1
        '#FFD1CF', '#F7A8A5', '#EF827E', '#E5655D',
        '#D85447', '#CB4635', '#BE3D23', '#AE2E11', '#9B1C00'
    ]
    cbar = colors.ListedColormap(cbar, name="blue_white_red_20")
    cbar.set_over('#641B00')
    cbar.set_under('#012A52')
    cbar.set_bad(color='white')

    try:
        field_to_plot = field['var']
    except:
        field_to_plot = field

    im = ax.contourf(field.lon, field.lat, field_to_plot,
                 levels=levels, transform=crs_latlon, cmap=cbar,
                     extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    cb.ax.tick_params(labelsize=8)
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    plt.show()

# ---------------------------------------------------------------------------- #
dmi_or = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
n34_or = Nino34CPC(xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc"),
    start=1920, end=2020)[0]

dmi = dmi_or.sel(time=slice('1951-01-01', '2016-12-01'))
n34 = n34_or.sel(time=slice('1951-01-01', '2016-12-01'))

n34_son = n34.sel(time=n34.time.dt.month.isin(10))
dmi_son = dmi.sel(time=dmi.time.dt.month.isin(10))

sst = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
sst = sst.drop_dims('nbnds')
sst = sst.sel(time=slice('1951-01-01', '2016-12-01')) # como en el paper
sst = sst.rename({'sst':'var'})

sst = xrFieldTimeDetrend(sst, 'time') # Detrend
sst = sst.rolling(time=3, center=True).mean() #estacional
sst = ((sst.groupby('time.month') - sst.groupby('time.month').mean('time'))
       /sst.groupby('time.month').std('time')) # standarizacion
sst = sst.sel(time=sst.time.dt.month.isin(10)) # SON
sst = sst.sel(month=10)

# ---------------------------------------------------------------------------- #
# Usando los ONI y DMI directamente
dmi_wo_n34_son, n34_wo_dmi_son = LinearReg1_D(
    n34_son.__mul__(1 / n34_son.std('time')),
    dmi_son.__mul__(1 / dmi_son.std('time')))

plotindex(n34_son, n34_wo_dmi_son, title='n34', label='residuo')
plotindex(dmi_son, dmi_wo_n34_son, title='dmi', label='residuo')

# eventos mas intensos
top_en, top_ln = find_top_events(n34_wo_dmi_son, 7)
top_iodp, top_iodn = find_top_events(dmi_wo_n34_son, 7)
# no hay gran coincidencia con lo obtenido en el paper a partir de las pcs

# Como se ven los campos reales de estos eventos?
top_en_events, top_ln_events = select_to_events(sst, top_en, top_ln)
top_iodp_events, top_iodn_events = select_to_events(sst, top_iodp, top_iodn)

# plot_sst_times(top_en_events)
# plot_sst_times(top_iodp_events)
# plot_sst_times(top_ln_events)
# plot_sst_times(top_iodn_events)
# no hay selección de eventos que en la realidad sean de signo opuesto,
# aunque algunos pueden estar cerca de ser neutros.
# sin embargo ninguno garantiza ser puro.

#  La descripcion de los residuos de las variables es muy vaga en el paper.
#  En los EOFs la selección de años es bastante similar a la que
#  obtienen en el paper.

# Campo que solo tenga N34, restandole la regresion del dmi sin n34.
sst_wo_dmi = sst - RegreField(sst, dmi_wo_n34_son)
sst_wo_n34 = sst - RegreField(sst, n34_wo_dmi_son)

top_en_reg_events, top_ln_reg_events = \
    select_to_events(sst_wo_dmi, top_en, top_ln)
top_iodp_reg_events, top_iodn_reg_events = \
    select_to_events(sst_wo_n34, top_iodp, top_iodn)

# plot_sst_times(top_en_reg_events)
# plot_sst_times(top_iodp_reg_events)
# plot_sst_times(top_ln_reg_events)
# plot_sst_times(top_iodn_reg_events)

# las composiciones dan muy mal. solo bien los positivos
PlotSSTOne(top_en_reg_events.mean('time'))
PlotSSTOne(top_iodp_reg_events.mean('time'))
PlotSSTOne(top_ln_reg_events.mean('time'))
PlotSSTOne(top_iodn_reg_events.mean('time'))

# EOF ------------------------------------------------------------------------ #
"""
Que pasa si en su lugar, los indices se definen a partir de EOF y luego se hace 
la regresion?
"""
# Esto puede dar lugar a diferencias: filtrado con harmonicos en lugar de Wavelet
filtered_sst = xr.apply_ufunc(
    bandpass_filter,
    sst['var'],
    7,                       # max_period
    input_core_dims=[['time'], [],],
    output_core_dims=[['time']],
    vectorize=True,
    dask=None,
    output_dtypes=[sst['var'].dtype],
    keep_attrs=True
)

filtered_sst = filtered_sst.transpose('time', 'lat', 'lon') # para Eof

# cuencas usadas en el paper
sst_pac = filtered_sst.sel(lon=slice(110,290), lat=slice(30,-30))
sst_ind = filtered_sst.sel(lon=slice(30,120), lat=slice(26,-26))

"""
Hipotesis: los eofs de cada cuenca son el mismo eof de las dos cuencas
"""
sst_indpac = filtered_sst.sel(lon=slice(30,290), lat=slice(25,-25))

# EOF pacifico - ENSO ---------------- #

pcscal = 1

solver_n34 = Eof(sst_pac)
eof_n34 = solver_n34.eofsAsCovariance(neofs=3)
pcs_n34 = solver_n34.pcs(pcscaling=pcscal)
var_per_n34 = np.around(solver_n34.varianceFraction(neigs=3).values * 100,1)

pc_n34 = pcs_n34.sel(mode=0) # indice ~n34

plt.imshow(-1*eof_n34[0], vmin=-1, vmax=1, cmap='RdBu_r')
plt.title(f'var exp {var_per_n34[0]}%')
plt.colorbar()
plt.show()

PlotBars(pc_n34)

plotindex(n34_son, -1*pc_n34, title='N34 EOF', label='EOF')

# EOF indico - IOD ---------------- #
solver_dmi = Eof(sst_ind)
eof_dmi = solver_dmi.eofsAsCovariance(neofs=3)
pcs_dmi = solver_dmi.pcs(pcscaling=pcscal)
var_per_dmi = np.around(solver_dmi.varianceFraction(neigs=3).values * 100, 1)

pc_dmi = pcs_dmi.sel(mode=0)# + pcs_dmi.sel(mode=1)

plt.imshow(-1*eof_dmi[0], vmin=-1, vmax=1, cmap='RdBu_r')
plt.title(f'var exp {var_per_dmi[0]}%')
plt.colorbar()
plt.show()
plotindex(dmi_son, -1*pc_dmi, title='DMI EOF', label='EOF')

PlotBars(pc_dmi)

# EOF pacifico-indico ---------------- #
solver_indpac = Eof(sst_indpac)
eof_indpac = solver_indpac.eofsAsCovariance(neofs=3)
pcs_indpac = solver_indpac.pcs(pcscaling=pcscal)
var_per = np.around(solver_indpac.varianceFraction(neigs=3).values * 100,1)

pc_indpac = pcs_indpac.sel(mode=0)

plt.imshow(-1*eof_indpac[0], vmin=-1, vmax=1, cmap='RdBu_r')
plt.colorbar()
plt.title(f'var exp {var_per[0]}%')
plt.show()

plotindex(n34_son, -1*pc_indpac, title='indpac EOF vs N34 ',
          label='EOF')
plotindex(dmi_son, -1*pc_indpac, title='indpac EOF vs DMI ',
          label='EOF')

# Como son estos indices frente a los otros? --------------------------------- #
corr_n34 = np.around(np.corrcoef(n34_son, -1*pc_n34)[0,1], 3)
corr_dmi = np.around(np.corrcoef(dmi_son, -1*pc_dmi)[0,1], 3)
corr_indpac = np.around(np.corrcoef(n34_son, -1*pc_indpac)[0,1], 3)
corr_indpac_dmi = np.around(np.corrcoef(dmi_son, -1*pc_indpac)[0,1], 3)
corr_n34_dmi = np.around(np.corrcoef(n34_son,dmi_son)[0,1], 3)
corr_pc_n34_dmi = np.around(np.corrcoef(-1*pc_n34, -1*pc_dmi)[0,1], 3)

print(f'Correlacion pc_n34 - n34: {corr_n34}')
print(f'Correlacion pc_dmi - dmi: {corr_dmi}')
print(f'Correlacion pc_indpac - n34: {corr_indpac}')
print(f'Correlacion pc_indpac - dmi: {corr_indpac_dmi}')
print(f'Correlacion n34 - dmi: {corr_n34_dmi}')
print(f'Correlacion pc_n34 - pc_dmi: {corr_pc_n34_dmi}')
# La correlacion entre los indces pc_dmi y pc_n34 es mas alta
# que la correlacion n34_son y dmi_son!!!

# ---------------------------------------------------------------------------- #
# Que pasa si se hace regresion con las pcs?
pc_dmi_wo_n34, pc_n34_wo_dmi = LinearReg1_D(-1*pc_n34, -1*pc_dmi)

top_en_pc, top_ln_pc = find_top_events(pc_n34_wo_dmi, 7)
top_iodp_pc, top_iodn_pc = find_top_events(pc_dmi_wo_n34, 7)
# Gran coincidencia de años seleccionados aca con los del paper

sst_wo_dmi_pc = filtered_sst - RegreField(filtered_sst, pc_dmi_wo_n34)
sst_wo_n34_pc = filtered_sst - RegreField(filtered_sst, pc_n34_wo_dmi)

top_en_pc_events, top_ln_pc_events = \
    select_to_events(sst_wo_dmi_pc, top_en_pc, top_ln_pc)
top_iodp_pc_events, top_iodn_pc_events = \
    select_to_events(sst_wo_n34_pc, top_iodp_pc, top_iodn_pc)

#plot_sst_times(top_en_pc_events)
# plot_sst_times(top_iodp_pc_events)
# plot_sst_times(top_ln_pc_events)
# plot_sst_times(top_iodn_pc_events)
# Al igual que antes los eventos no son puros

# Las composiciones dan bastante bien
PlotSSTOne(top_en_pc_events.mean('time'), levels=np.arange(-0.9,1,0.1))
PlotSSTOne(top_iodp_pc_events.mean('time'), levels=np.arange(-0.9,1,0.1))
PlotSSTOne(top_ln_pc_events.mean('time'), levels=np.arange(-0.9,1,0.1))
PlotSSTOne(top_iodn_pc_events.mean('time'), levels=np.arange(-0.9,1,0.1))
# ---------------------------------------------------------------------------- #