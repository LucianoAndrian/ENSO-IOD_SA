"""
Testeo replica kayano con regresion falopa
"""
"""
Antes de usar eof, total es casi lo mismo.
¿los composites sinteticos luego de la regresion pueden contener años que en
 la realidad son opuestos?
"""
import statsmodels.formula.api as smf
import pandas as pd
import matplotlib.pyplot as plt
from eofs.xarray import Eof
import numpy as np
import xarray as xr
from Funciones import DMI, Nino34CPC, ComputeWithEffect, ComputeWithoutEffect
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

def RegWEffect(n34, dmi,data=None, data2=None, m=9,two_variables=False):
    var_reg_n34_2=0
    var_reg_dmi_2=1

    data['time'] = n34
     #print('Full Season')
    # try:
    #     aux = LinearReg(data.groupby('month')[m], 'time')
    # except:
    #     aux = LinearReg(data.groupby('time.month')[m], 'time')
    aux = LinearReg(data, 'time')
    var_reg_n34 = xr.polyval(data.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
           aux.var_polyfit_coefficients[1]
    #var_reg_n34 = aux.var_polyfit_coefficients[0]

    data['time'] = dmi
    # try:
    #     aux = LinearReg(data.groupby('month')[m], 'time')
    # except:
    #     aux = LinearReg(data.groupby('time.month')[m], 'time')
    aux = LinearReg(data, 'time')
    var_reg_dmi = xr.polyval(data.groupby('month')[m].time, aux.var_polyfit_coefficients[0]) + \
           aux.var_polyfit_coefficients[1]
    # var_reg_dmi = aux.var_polyfit_coefficients[0]
    # var_reg_dmi = aux.var_polyfit_coefficients[0]

    if two_variables:
        print('Two Variables')

        data2['time'] = n34
        #print('Full Season data2, m ignored')
        #aux = LinearReg(data2.groupby('month')[m], 'time')
        aux = LinearReg(data2, 'time')
        var_reg_n34_2 = aux.var_polyfit_coefficients[0]

        data2['time'] = dmi
        #aux = LinearReg(data2.groupby('month')[m], 'time')
        aux = LinearReg(data2, 'time')
        var_reg_dmi_2 = aux.var_polyfit_coefficients[0]

    return var_reg_n34, var_reg_dmi, var_reg_n34_2, var_reg_dmi_2

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

"""
Regresion parcial observada
"""


# ---------------------------------------------------------------------------- #
data_dir_t_pp = '/pikachu/datos/luciano.andrian/observado/ncfiles/' \
                'data_obs_d_w_c/' #T y PP ya procesados
data_dir = '/pikachu/datos/luciano.andrian/observado/ncfiles/ERA5/1950_2020/'

# ---------------------------------------------------------------------------- #
dmi_or = DMI(filter_bwa=False, start_per='1920', end_per='2020')[2]
n34_or = Nino34CPC(xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc"),
    start=1920, end=2020)[0]

print('N34, DMI ------------------------------------------------------------ #')
dmi = dmi_or.sel(time=slice('1951-01-01', '2016-12-01'))
n34 = n34_or.sel(time=slice('1951-01-01', '2016-12-01'))

# Regrsion solo en SON:
n34_son = n34.sel(time=n34.time.dt.month.isin(10))
dmi_son = dmi.sel(time=dmi.time.dt.month.isin(10))
dmi_wo_n34_son, n34_wo_dmi_son = LinearReg1_D(
    n34_son.__mul__(1 / n34_son.std('time')),
    dmi_son.__mul__(1 / dmi_son.std('time')))

# la perdida de magnitud es mas importante
plotindex(n34_son, n34_wo_dmi_son, title='n34', label='residuo')
plotindex(dmi_son, dmi_wo_n34_son, title='dmi', label='residuo')

"""
Pero no es posible detectar como Niño una Niña en eventos que no sean cercanos 
a neutro

Con el DMI puede pasar cuando la magnitud es menor, pero cno en los mas fuertes
"""

# Que pasa si en lugar de los indices se usa el metodo de eof que usaron en el paper?
def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt

sst = xr.open_dataset(
    "/pikachu/datos/luciano.andrian/verif_2019_2023/sst.mnmean.nc")
sst = sst.drop_dims('nbnds')
sst = sst.sel(time=slice('1951-01-01', '2016-12-01')) # como en el paper
sst = sst.rename({'sst':'var'})

sst = xrFieldTimeDetrend(sst, 'time')
sst = sst.rolling(time=3, center=True).mean()
sst = sst.groupby('time.month') - sst.groupby('time.month').mean('time')
sst = sst.sel(time=sst.time.dt.month.isin(10))
sst = sst/sst.std('time')
# aca faltaria el filtro morlet de 2-7 años... mucho lio para esto..

import numpy as np

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

filtered_sst = xr.apply_ufunc(
    bandpass_filter,
    sst['var'],
    7,                       # max_period
    input_core_dims=[['time'], [],],
    output_core_dims=[['time']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[sst['var'].dtype],
    keep_attrs=True
)

filtered_sst = filtered_sst.transpose('time', 'lat', 'lon')

# cuencas usadas en el paper
sst_pac = filtered_sst.sel(lon=slice(110,290), lat=slice(30,-30))
sst_ind = filtered_sst.sel(lon=slice(30,120), lat=slice(26,-26))

# hipotesis, tomar eof por separado no separa nada.
# Solo identifica la misma variabilidad en cada cuenca
sst_pacind = filtered_sst.sel(lon=slice(30,290), lat=slice(25,-25))

# EOF pacifico - ENSO
solver_n34 = Eof(sst_pac)
eof_n34 = solver_n34.eofsAsCovariance(neofs=3)
pcs_n34 = solver_n34.pcs()
var_per_n34 = np.around(solver_n34.varianceFraction(neigs=3).values * 100,1)

pc_n34 = pcs_n34.sel(mode=0) # indice ~n34

plt.imshow(-1*eof_n34[0], vmin=-1, vmax=1, cmap='RdBu_r')
plt.title(f'var exp {var_per_n34[0]}%')
plt.colorbar()
plt.show()

values = -1*pc_n34/pc_n34.std()
years = pc_n34.time.dt.year
# Colores: azul si valor > 0, rojo si valor < 0
colors = ['blue' if v > 0 else 'red' for v in values]
plt.figure(figsize=(10, 4))
plt.bar(x=years, height=values, color=colors)
plt.grid(True)
plt.ylim((-2, 3))
plt.xlabel("Año")
plt.show()

plotindex(n34_son, -1*pc_n34/pc_n34.std(), title='N34 EOF', label='EOF')

# EOF indico - IOD
solver_dmi = Eof(sst_ind)
eof_dmi = solver_dmi.eofsAsCovariance(neofs=3)
pcs_dmi = solver_dmi.pcs()
var_per_dmi = np.around(solver_dmi.varianceFraction(neigs=3).values * 100, 1)

pc_dmi = pcs_dmi.sel(mode=0)# + pcs_dmi.sel(mode=1)

plt.imshow(-1*eof_dmi[0], vmin=-1, vmax=1, cmap='RdBu_r')
plt.title(f'var exp {var_per_dmi[0]}%')
plt.colorbar()
plt.show()
plotindex(dmi_son, -1*pc_dmi/pc_dmi.std(), title='DMI EOF', label='EOF')

values = -1 * pc_dmi / pc_dmi.std()
years = pc_dmi.time.dt.year
# Colores: azul si valor > 0, rojo si valor < 0
colors = ['blue' if v > 0 else 'red' for v in values]
plt.figure(figsize=(10, 4))
plt.bar(x=years, height=values, color=colors)
plt.grid(True)
plt.ylim((-2, 3))
plt.xlabel("Año")
plt.show()



# EOF pacifico-indico
solver_pacind = Eof(sst_pacind)
eof_pacind = solver_pacind.eofsAsCovariance(neofs=3)
pcs_pacind = solver_pacind.pcs()
var_per = np.around(solver_pacind.varianceFraction(neigs=3).values * 100,1)

pc_pacind = pcs_pacind.sel(mode=0)

plt.imshow(-1*eof_pacind[0], vmin=-1, vmax=1, cmap='RdBu_r')
plt.colorbar()
plt.title(f'var exp {var_per[0]}%')
plt.show()
#oh, wow, es lo mismo en cada cuenca...

plotindex(n34_son, -1*pc_pacind/pc_pacind.std(), title='indpac EOF vs N34 ',
          label='EOF')
plotindex(dmi_son, -1*pc_pacind/pc_pacind.std(), title='indpac EOF vs DMI ',
          label='EOF')


corr_n34 = np.around(np.corrcoef(
    n34.sel(time=n34.time.dt.month.isin(10)), -1*pc_n34)[0,1], 3)
corr_dmi = np.around(np.corrcoef(
    dmi.sel(time=dmi.time.dt.month.isin(10)), -1*pc_dmi)[0,1], 3)
corr_pacind = np.around(np.corrcoef(
    n34.sel(time=n34.time.dt.month.isin(10)), -1*pc_pacind)[0,1], 3)
corr_pacind_dmi = np.around(np.corrcoef(
    dmi.sel(time=dmi.time.dt.month.isin(10)), -1*pc_pacind)[0,1], 3)

print(f'Correlacion pc_n34 - n34: {corr_n34}')
print(f'Correlacion pc_dmi - dmi: {corr_dmi}')
print(f'Correlacion pc_pacind - n34: {corr_pacind}')
print(f'Correlacion pc_pacind - dmi: {corr_pacind_dmi}')


# Que pasa si se hace regresion con las pcs?
pc_dmi_wo_n34, pc_n34_wo_dmi = LinearReg1_D(
    -1*pc_n34.__mul__(1 / pc_n34.std('time')),
    -1*pc_dmi.__mul__(1 / pc_dmi.std('time')))

plotindex(n34.sel(time=n34.time.dt.month.isin(10)), pc_n34_wo_dmi,
          title='N34 EOF regre', label='EOF')
plotindex(dmi.sel(time=dmi.time.dt.month.isin(10)), pc_dmi_wo_n34,
          title='DMI EOF regre', label='EOF')

"""
Ahora hay diferencias notables entre los indices y las pcs
"""

# Tomando los 7 eventos mas intensos
#EN puros mas intensos - pc_index
top_en_pc_index = pc_n34_wo_dmi.sortby(
    pc_n34_wo_dmi, ascending=False).isel(time=slice(10)).time.values
#LN puros mas intensos - pc_index
top_ln_pc_index = pc_n34_wo_dmi.sortby(
    pc_n34_wo_dmi, ascending=True).isel(time=slice(10)).time.values

#IODp puros mas intensos - pc_index
top_iodp_pc_index = pc_dmi_wo_n34.sortby(
    pc_dmi_wo_n34, ascending=False).isel(time=slice(10)).time.values
#IODn puros mas intensos - pc_index
top_iodn_pc_index = pc_dmi_wo_n34.sortby(
    pc_dmi_wo_n34, ascending=True).isel(time=slice(10)).time.values


"""
Ahora a partir de los residuos de las pcs, buscar los residuos de las variables
a usar, SST en este caso.
"""

filtered_sst = filtered_sst.to_dataset(name="var")
aux_n34_wodmi, aux_dmi_won34, data_n34_wodmi, data_dmi_won34 = \
    RegWOEffect(n34=pc_n34.__mul__(1 / pc_n34.std('time')),
                n34_wo_dmi=pc_n34_wo_dmi,
                dmi=pc_dmi.__mul__(1 / pc_dmi.std('time')),
                dmi_wo_n34=pc_dmi_wo_n34,
                m=10, datos=filtered_sst/filtered_sst.std('time'))

# aux_n34, aux_dmi, aux_n34_2, aux_dmi_2 = \
#     RegWEffect(data=filtered_sst/filtered_sst.std('time'),
#                n34=pc_n34.__mul__(1 / pc_n34.std('time')),
#                dmi=pc_dmi.__mul__(1 / pc_dmi.std('time')),
#                m=10, two_variables=False)
#
# aux_dmi_won34 = aux_dmi - aux_n34
# aux_n34_wodmi = aux_n34 - aux_dmi

# Seleccionando de los residuos de SST en cada caso, los eventos mas fuertes
data_n34_wodmi['time'] = pc_n34.time.values
data_n34_wodmi = data_n34_wodmi/data_n34_wodmi.std('time')
events_top_en_pc_index = data_n34_wodmi.sel(time=top_en_pc_index)
events_top_ln_pc_index = data_n34_wodmi.sel(time=top_ln_pc_index)

data_dmi_won34['time'] = pc_dmi.time.values
data_dmi_won34 = data_dmi_won34/data_dmi_won34.std('time')
events_top_iodp_pc_index = data_dmi_won34.sel(time=top_iodp_pc_index)
events_top_iodn_pc_index = data_dmi_won34.sel(time=top_iodn_pc_index)

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

# Ploteo de todos los eventos seleccionados y el año
plot_sst_times(events_top_en_pc_index) # EN
plot_sst_times(events_top_ln_pc_index) # LN
plot_sst_times(events_top_iodp_pc_index) # IODp
plot_sst_times(events_top_iodn_pc_index) # IODn



plt.imshow(events_top_en_pc_index.mean('time')['var'],vmin=-2, vmax=2, cmap='RdBu_r');plt.show()
plt.imshow(events_top_ln_pc_index.mean('time')['var'],vmin=-2, vmax=2, cmap='RdBu_r');plt.show()
plt.imshow(events_top_iodp_pc_index.mean('time')['var'],vmin=-2, vmax=2, cmap='RdBu_r');plt.show()
plt.imshow(events_top_iodn_pc_index.mean('time')['var'],vmin=-2, vmax=2, cmap='RdBu_r');plt.show()


