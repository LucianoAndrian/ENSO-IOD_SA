"""
Funciones generales
"""
# ---------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import logging
import os
from scipy.integrate import trapz
import scipy.stats as st
import pandas as pd

# ---------------------------------------------------------------------------- #
def xrFieldTimeDetrend(xrda, dim, deg=1):
    # detrend along a single dimension
    aux = xrda.polyfit(dim=dim, deg=deg)
    try:
        trend = xr.polyval(xrda[dim], aux.var_polyfit_coefficients[0])
    except:
        trend = xr.polyval(xrda[dim], aux.polyfit_coefficients[0])

    dt = xrda - trend
    return dt

# ---------------------------------------------------------------------------- #
def RegreField(field, index, return_coef=False):
    """
    Devuelve la parte del campo `field` explicada linealmente por `index`.
    """
    if isinstance(field, xr.Dataset):
        da = field['var']
    else:
        da = field

    # 2 usar el indice en "time" para usar esa dimencion para la regresion
    da_idx = da.copy()
    da_idx = da_idx.assign_coords(time=index)

    # 3 Regresión
    coef = da_idx.polyfit(dim='time', deg=1, skipna=True).polyfit_coefficients
    beta      = coef.sel(degree=1)   # pendiente
    intercept = coef.sel(degree=0)   # término independiente

    # 4 Reconstruir la parte explicada y restaurar las fechas reales
    fitted = beta * da_idx['time'] + intercept
    fitted = fitted.assign_coords(time=da['time'])

    if return_coef is True:
        result = beta
    else:
        result = fitted

    return result

# ---------------------------------------------------------------------------- #
def open_and_load(path):
    ds = xr.open_dataset(path, engine='netcdf4')  # backend explícito
    ds_loaded = ds.load()  # carga a memoria
    ds.close()             # cierra archivo en disco
    return ds_loaded

# ---------------------------------------------------------------------------- #
def MakeMask(DataArray, dataname='mask'):
    import regionmask
    mask=regionmask.defined_regions.natural_earth_v5_0_0.countries_110.mask(DataArray)
    mask = xr.where(np.isnan(mask), mask, 1)
    mask = mask.to_dataset(name=dataname)
    return mask

# ---------------------------------------------------------------------------- #
def SameDateAs(data, datadate):
    """
    En data selecciona las mismas fechas que datadate
    :param data:
    :param datadate:
    :return:
    """
    return data.sel(time=datadate.time.values)

# ---------------------------------------------------------------------------- #
def MakerMaskSig(data, r_crit):
    mask_sig = data.where((data < -1 * r_crit) | (data > r_crit))
    mask_sig = mask_sig.where(np.isnan(mask_sig), 1)

    return mask_sig

# ---------------------------------------------------------------------------- #
def Weights(data):
    weights = np.transpose(np.tile(np.cos(data.lat * np.pi / 180),
                                   (len(data.lon), 1)))
    data_w = data * weights
    return data_w

# ---------------------------------------------------------------------------- #
def init_logger(log_name="app.log", level=logging.INFO):
    """
    Inicializa un logger que guarda el log en una carpeta 'logs'
    ubicada en el mismo directorio del script que lo llama.
    """
    # directorio donde está ESTE archivo
    # y subir un nivel -> directorio raíz del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

# ---------------------------------------------------------------------------- #
def ChangeLons(data, lon_name='lon'):
    data['_longitude_adjusted'] = xr.where(
        data[lon_name] < 0,
        data[lon_name] + 360,
        data[lon_name])

    data = (
        data
            .swap_dims({lon_name: '_longitude_adjusted'})
            .sel(**{'_longitude_adjusted': sorted(data._longitude_adjusted)})
            .drop(lon_name))

    data = data.rename({'_longitude_adjusted': 'lon'})

    return data

# ---------------------------------------------------------------------------- #
def AreaBetween(curva1, curva2):
    """
    Calcula el área entre dos curvas dadas como pandas.Series,

    Parámetros:
    curva1 : pd.Series -> Primera curva (índices = x, valores = densidad)
    curva2 : pd.Series -> Segunda curva (índices = x, valores = densidad)

    Retorna:
    float -> Área entre las dos curvas
    """
    # Extraer valores x e y de cada curva
    x = curva1.index.values
    y1 = curva1.values
    y2 = curva2.values

    max1 = curva1.idxmax()
    max2 = curva2.idxmax()

    sign = -1*np.sign(max1 - max2)
    if sign == 0:
        sign = 1
    # Calcular el área entre curvas
    area_between = trapz(np.abs(y1 - y2), x)
    if area_between <0: # pasa por el orden de x
        area_between = -1* area_between

    return area_between*sign

# ---------------------------------------------------------------------------- #
def RenameDataset(new_name, *args):
    dataset = []
    for arg in args:
        arg2 = arg.rename({list(arg.data_vars)[0]:new_name})

        dataset.append(arg2)

    return tuple(dataset)

# ---------------------------------------------------------------------------- #
def pdf_fit_normal(data, size, start, end):
    distribution = st.norm
    params = distribution.fit(data)
    # parametros
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    x = np.linspace(start, end, size)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# ---------------------------------------------------------------------------- #
def PDF_cases(variable, season, box_lons, box_lats, box_name,
              cases, cases_dir):

    if variable == 'prec':
        fix_factor = 30
    elif variable == 'tref':
        fix_factor = 1
    else:
        fix_factor = 9.8

    # climatologia
    neutro = xr.open_dataset(
        f'{cases_dir}{variable}_neutros_{season}_detrend_05.nc')
    neutro = neutro.rename({list(neutro.data_vars)[0]: 'var'})
    neutro = neutro * fix_factor

    mask_ocean = MakeMask(neutro, list(neutro.data_vars)[0])
    neutro = neutro * mask_ocean

    resultados_regiones = {}
    for bl, bt, name in zip(box_lons, box_lats, box_name):
        aux_neutro = neutro.sel(lon=slice(bl[0], bl[1]),
                                lat=slice(bt[0], bt[1]))
        clim = aux_neutro.mean(['lon', 'lat'])

        aux_resultados = {}
        if variable == 'prec':
            if name == 'Patagonia':
                startend = -30
            elif name == 'N-SESA':
                startend = -80
            else:
                startend = -60
        else:
            startend = -2

        aux_clim = clim - clim.mean('time')
        aux_clim = np.nan_to_num(aux_clim['var'].values)

        pdf_clim_full = pdf_fit_normal(aux_clim, 500,
                                       -1 * startend, startend)

        aux_resultados['clim'] = pdf_clim_full

        for c_count, c in enumerate(cases):
            case = xr.open_dataset(
                f'{cases_dir}{variable}_{c}_{season}_detrend_05.nc')
            case = case.rename({list(case.data_vars)[0]: 'var'})
            case = case * fix_factor
            mask_ocean = MakeMask(case, list(case.data_vars)[0])
            case = case * mask_ocean
            case = case.sel(lon=slice(bl[0], bl[1]), lat=slice(bt[0], bt[1]))
            case = case.mean(['lon', 'lat'])

            case_anom = case - clim.mean('time')
            case_anom = np.nan_to_num(case_anom['var'].values)

            pdf_case = pdf_fit_normal(case_anom, 500,
                                      -startend,  startend)

            aux_resultados[c] = pdf_case

        resultados_regiones[name] = aux_resultados

    return resultados_regiones

# ---------------------------------------------------------------------------- #
