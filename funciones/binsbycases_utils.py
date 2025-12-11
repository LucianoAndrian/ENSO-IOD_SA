
import numpy as np
import xarray as xr
from funciones.general_utils import MakeMask
from funciones.selectvariables_utils import SelectVariables

# ---------------------------------------------------------------------------- #
def SelectBins2D(serie, bins_limits):

    bins = []
    bins_limits = [[bins_limits[i], bins_limits[i+1]]
                   for i in range(len(bins_limits)-1)]

    for bl in bins_limits:
        bmin = bl[0]
        bmax = bl[1]
        x = serie.where(serie >= bmin)
        x = x.where(x < bmax)
        bins.append(x)

    bins = xr.concat(bins, dim='bins')

    return bins

# ---------------------------------------------------------------------------- #
def SelectDatesBins(bins, bin_data, min_percentage=0.1):

    bins_name_var = list(bins.data_vars)[0]
    bin_data_name_var = list(bin_data.data_vars)[0]

    bin_data_f = []
    bin_len = []
    for bl in bins.bins.values:
        aux_bins = bins.sel(bins=bl)

        bins_r = []
        bin_len_r = []
        for r in bins.r.values:
            aux = aux_bins.sel(r=r)
            bin_len_r.append(len(aux.where(
                ~np.isnan(aux.sst), drop=True)[bins_name_var]))
            bin_data_sel = bin_data.sel(r=r)
            dates_bins = aux.time[np.where(~np.isnan(aux[bins_name_var]))]
            bin_data_sel = bin_data_sel.sel(
                time=bin_data_sel.time.isin(dates_bins))
            bins_r.append(bin_data_sel)

        bin_len.append(np.sum(bin_len_r))
        bins_r_f = xr.concat(bins_r, dim='r')
        bin_data_f.append(bins_r_f)

    bin_data_f = xr.concat(bin_data_f, dim='bins')

    bin_data_mean = bin_data_f.mean(['r', 'time'], skipna=True)
    bin_data_mean = list(bin_data_mean[bin_data_name_var].values)

    bin_data_std = bin_data_f.std(['r', 'time'], skipna=True)
    bin_data_std = list(bin_data_std[bin_data_name_var].values)

    check = sum(bin_len)*min_percentage

    bin_data_mean, bin_data_std = zip(*[
        (0 if count < check else mean,
         0 if count < check else std)
        for count, mean, std in
        zip(bin_len, bin_data_mean, bin_data_std)
    ])

    return bin_data_mean, bin_data_std, bin_len

# ---------------------------------------------------------------------------- #
def SetBinsByCases(indices, magnitudes, bin_limits, cases):

    mt_dim = (len(magnitudes) * 2 + 1)

    matriz_base = np.full((mt_dim, mt_dim), None, dtype=object)

    columna_fila_centro = {}
    for i, u in zip(indices, ['columna_centro', 'fila_centro']):
        aux_names = []

        for signo in ['n', 'p']:
            if signo == 'p':
                aux_magnitudes = magnitudes[::-1]
            else:
                aux_magnitudes = magnitudes

            for m in aux_magnitudes:
                aux_names.append(f'{m}_{i}{signo}')
            if signo == 'n':
                aux_names.append('clim')

        columna_fila_centro[u] = aux_names

    matriz_base[:, mt_dim // 2] = columna_fila_centro['columna_centro'][::-1]
    matriz_base[mt_dim // 2, :] = columna_fila_centro['fila_centro']

    # combinaciones
    for i in range(mt_dim):
        if i != mt_dim // 2:
            for j in range(mt_dim):
                if j != mt_dim // 2:
                    matriz_base[i, j] = \
                        (f'{matriz_base[:, mt_dim // 2][i]}-'
                         f'{matriz_base[mt_dim // 2, :][j]}')

    cases_magnitude = matriz_base.flatten().tolist()

    bins_limits_pos = []
    bins_limits_neg = []
    bins_limits_neutro = []
    for bl_count, bl in enumerate(bin_limits):
        if sum(bl) > 0:
            bins_limits_pos.append(bl_count)
        elif sum(bl) < 0:
            bins_limits_neg.append(bl_count)
        elif sum(bl) == 0:
            bins_limits_neutro.append(bl_count)

    bins_by_cases_indice1 = []
    bins_by_cases_indice2 = []
    for c in cases:
        check_pos_neg = True

        if (indices[0] in c) and ('puros' in c):
            bins_by_cases_indice2.append(bins_limits_neutro)
        if (indices[1] in c) and ('puros' in c):
            bins_by_cases_indice1.append(bins_limits_neutro)

        if 'neutros' in c:
            bins_by_cases_indice1.append(bins_limits_neutro)
            bins_by_cases_indice2.append(bins_limits_neutro)

        if (f'{indices[0]}_pos' in c) and (f'{indices[1]}_neg' in c):
            bins_by_cases_indice1.append(bins_limits_pos)
            bins_by_cases_indice2.append(bins_limits_neg)
            check_pos_neg = False
        elif (f'{indices[0]}_neg' in c) and (f'{indices[1]}_pos' in c):
            bins_by_cases_indice1.append(bins_limits_neg)
            bins_by_cases_indice2.append(bins_limits_pos)
            check_pos_neg = False

        if (indices[0] in c) and ('pos' in c) and check_pos_neg:
            bins_by_cases_indice1.append(bins_limits_pos)
        elif (indices[0] in c) and ('neg' in c) and check_pos_neg:
            bins_by_cases_indice1.append(bins_limits_neg)

        if (indices[1] in c) and ('pos' in c) and check_pos_neg:
            bins_by_cases_indice2.append(bins_limits_pos)
        elif (indices[1] in c) and ('neg' in c) and check_pos_neg:
            bins_by_cases_indice2.append(bins_limits_neg)

        if ('sim' in c) and ('pos' in c):
            bins_by_cases_indice1.append(bins_limits_pos)
            bins_by_cases_indice2.append(bins_limits_pos)
        elif ('sim' in c) and ('neg' in c):
            bins_by_cases_indice1.append(bins_limits_neg)
            bins_by_cases_indice2.append(bins_limits_neg)

    bin_names = magnitudes + [''] + magnitudes[::-1]

    cases_names = []
    for c_count, c in enumerate(cases):
        aux_h = '-'
        for i1 in bins_by_cases_indice1[c_count]:
            i1_aux = sum(bin_limits[i1])
            i1_aux_mag_name = bin_names[i1]
            i1_aux_h = '_'
            if i1_aux > 0:
                i1_aux_name = indices[0] + 'p'
            elif i1_aux < 0:
                i1_aux_name = indices[0] + 'n'
            elif i1_aux == 0:
                i1_aux_name = ''
                i1_aux_mag_name = ''
                i1_aux_h = ''
                aux_h = ''

            i1_name = f"{i1_aux_mag_name}{i1_aux_h}{i1_aux_name}"

            for i2 in bins_by_cases_indice2[c_count]:
                i2_aux = sum(bin_limits[i2])
                i2_aux_mag_name = bin_names[i2]
                i2_aux_h = '_'

                if i2_aux > 0:
                    i2_aux_name = indices[1] + 'p'
                elif i2_aux < 0:
                    i2_aux_name = indices[1] + 'n'
                elif i2_aux == 0:
                    i2_aux_name = ''
                    i2_aux_mag_name = ''
                    i2_aux_h = ''
                    aux_h = ''

                i2_name = f"{i2_aux_mag_name}{i2_aux_h}{i2_aux_name}"
                case_name = f"{i1_name}{aux_h}{i2_name}"
                cases_names.append(case_name)

    return cases_names, cases_magnitude, \
        bins_by_cases_indice1, bins_by_cases_indice2

# ---------------------------------------------------------------------------- #
def SelectBins(data, min, max, sd=1):
    # sd opcional en caso de no estar escalado
    if np.abs(min) > np.abs(max):
        return (data >= min*sd) & (data < max*sd)
    elif np.abs(min) < np.abs(max):
        return (data > min*sd) & (data <= max*sd)
    elif np.abs(min) == np.abs(max):
        return (data >= min*sd) & (data <= max*sd)

# ---------------------------------------------------------------------------- #
def SetBinsByCases(indices, magnitudes, bin_limits, cases):

    mt_dim = (len(magnitudes) * 2 + 1)

    matriz_base = np.full((mt_dim, mt_dim), None, dtype=object)

    columna_fila_centro = {}
    for i, u in zip(indices, ['columna_centro', 'fila_centro']):
        aux_names = []

        for signo in ['n', 'p']:
            if signo == 'p':
                aux_magnitudes = magnitudes[::-1]
            else:
                aux_magnitudes = magnitudes

            for m in aux_magnitudes:
                aux_names.append(f'{m}_{i}{signo}')
            if signo == 'n':
                aux_names.append('clim')

        columna_fila_centro[u] = aux_names

    matriz_base[:, mt_dim // 2] = columna_fila_centro['columna_centro'][::-1]
    matriz_base[mt_dim // 2, :] = columna_fila_centro['fila_centro']

    # combinaciones
    for i in range(mt_dim):
        if i != mt_dim // 2:
            for j in range(mt_dim):
                if j != mt_dim // 2:
                    matriz_base[i, j] = \
                        (f'{matriz_base[:, mt_dim // 2][i]}-'
                         f'{matriz_base[mt_dim // 2, :][j]}')

    cases_magnitude = matriz_base.flatten().tolist()

    bins_limits_pos = []
    bins_limits_neg = []
    bins_limits_neutro = []
    for bl_count, bl in enumerate(bin_limits):
        if sum(bl) > 0:
            bins_limits_pos.append(bl_count)
        elif sum(bl) < 0:
            bins_limits_neg.append(bl_count)
        elif sum(bl) == 0:
            bins_limits_neutro.append(bl_count)

    bins_by_cases_indice1 = []
    bins_by_cases_indice2 = []
    for c in cases:
        check_pos_neg = True

        if (indices[0] in c) and ('puros' in c):
            bins_by_cases_indice2.append(bins_limits_neutro)
        if (indices[1] in c) and ('puros' in c):
            bins_by_cases_indice1.append(bins_limits_neutro)

        if 'neutros' in c:
            bins_by_cases_indice1.append(bins_limits_neutro)
            bins_by_cases_indice2.append(bins_limits_neutro)

        if (f'{indices[0]}_pos' in c) and (f'{indices[1]}_neg' in c):
            bins_by_cases_indice1.append(bins_limits_pos)
            bins_by_cases_indice2.append(bins_limits_neg)
            check_pos_neg = False
        elif (f'{indices[0]}_neg' in c) and (f'{indices[1]}_pos' in c):
            bins_by_cases_indice1.append(bins_limits_neg)
            bins_by_cases_indice2.append(bins_limits_pos)
            check_pos_neg = False

        if (indices[0] in c) and ('pos' in c) and check_pos_neg:
            bins_by_cases_indice1.append(bins_limits_pos)
        elif (indices[0] in c) and ('neg' in c) and check_pos_neg:
            bins_by_cases_indice1.append(bins_limits_neg)

        if (indices[1] in c) and ('pos' in c) and check_pos_neg:
            bins_by_cases_indice2.append(bins_limits_pos)
        elif (indices[1] in c) and ('neg' in c) and check_pos_neg:
            bins_by_cases_indice2.append(bins_limits_neg)

        if ('sim' in c) and ('pos' in c):
            bins_by_cases_indice1.append(bins_limits_pos)
            bins_by_cases_indice2.append(bins_limits_pos)
        elif ('sim' in c) and ('neg' in c):
            bins_by_cases_indice1.append(bins_limits_neg)
            bins_by_cases_indice2.append(bins_limits_neg)

    bin_names = magnitudes + [''] + magnitudes[::-1]

    cases_names = []
    for c_count, c in enumerate(cases):
        aux_h = '-'
        for i1 in bins_by_cases_indice1[c_count]:
            i1_aux = sum(bin_limits[i1])
            i1_aux_mag_name = bin_names[i1]
            i1_aux_h = '_'
            if i1_aux > 0:
                i1_aux_name = indices[0] + 'p'
            elif i1_aux < 0:
                i1_aux_name = indices[0] + 'n'
            elif i1_aux == 0:
                i1_aux_name = ''
                i1_aux_mag_name = ''
                i1_aux_h = ''
                aux_h = ''

            i1_name = f"{i1_aux_mag_name}{i1_aux_h}{i1_aux_name}"

            for i2 in bins_by_cases_indice2[c_count]:
                i2_aux = sum(bin_limits[i2])
                i2_aux_mag_name = bin_names[i2]
                i2_aux_h = '_'

                if i2_aux > 0:
                    i2_aux_name = indices[1] + 'p'
                elif i2_aux < 0:
                    i2_aux_name = indices[1] + 'n'
                elif i2_aux == 0:
                    i2_aux_name = ''
                    i2_aux_mag_name = ''
                    i2_aux_h = ''
                    aux_h = ''

                i2_name = f"{i2_aux_mag_name}{i2_aux_h}{i2_aux_name}"
                case_name = f"{i1_name}{aux_h}{i2_name}"
                cases_names.append(case_name)

    return cases_names, cases_magnitude, \
        bins_by_cases_indice1, bins_by_cases_indice2

# ---------------------------------------------------------------------------- #
def BinsByCases_noComp(v, v_name, fix_factor, s, mm, c, c_count,
                       bin_limits, bins_by_cases_dmi, bins_by_cases_n34,
                       dates_dir, cases_dir, snr=False, neutro_clim=False,
                       box=False, box_lat=[], box_lon=[], ocean_mask=False):

    def Weights(data):
        weights = np.transpose(
            np.tile(np.cos(np.arange(
                data.lat.min().values,
                data.lat.max().values + 1) * np.pi / 180),
                    (len(data.lon), 1)))
        try:
            data_w = data * weights
        except:
            data_w = data.transpose('time', 'lat', 'lon') * weights

        return data_w

    # 1. se abren los archivos de los índices (completos y se pesan por su SD)
    # tambien los archivos de la variable en cuestion pero para cada "case" = c

    data_dates_dmi_or = xr.open_dataset(dates_dir + 'DMI_' + s +
                                        '_Leads_r_CFSv2.nc')
    data_dates_dmi_or /= data_dates_dmi_or.mean('r').std()

    data_dates_n34_or = xr.open_dataset(dates_dir + 'N34_' + s +
                                        '_Leads_r_CFSv2.nc')

    aux_n34_std = data_dates_n34_or.mean('r').std()
    data_dates_n34_or /= aux_n34_std

    print('1.1 Climatología y case')
    if v != 'hgt':
        end_nc_file = '_detrend_05.nc'
        if v == 'hgt750': # que maravilla...
            end_nc_file = '__detrend_05.nc'
        #end_nc_file = '_no_detrend_05.nc'

    else:
        end_nc_file = '_05.nc'

    if neutro_clim:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_neutros' + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)
    else:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_' + s.lower() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)

    if ocean_mask is True:
        mask = MakeMask(clim, list(clim.data_vars)[0])
        clim = clim * mask

    try:
        case = Weights(
            xr.open_dataset(cases_dir + v + '_' + c + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)

        if ocean_mask is True:
            mask = MakeMask(case, list(case.data_vars)[0])
            case = case * mask

    except:
        print(f"case {c}, no encontrado para {v}")
        aux = clim.mean('time').__mul__(0)
        return aux, aux, aux

    try:
        clim = clim.sel(P=750)
        case = case.sel(P=750)
    except:
        pass

    if v == 'tref' or v == 'prec' or v == 'tsigma':
        lat = np.arange(-60, 20 + 1)
        lon = np.arange(275, 330 + 1)
    else:
        lat = np.linspace(-80, 20, 101)
        lon = np.linspace(0, 359, 360)

    if clim.lat[0] > clim.lat[-1]:
        lat_clim = lat[::-1]
    else:
        lat_clim = lat
    clim = clim.sel(lat=slice(lat_clim[0], lat_clim[-1]),
                    lon=slice(lon[0], lon[-1]))

    if case.lat[0] > case.lat[-1]:
        lat_case = lat[::-1]
    else:
        lat_case = lat
    case = case.sel(lat=slice(lat_case[0], lat_case[-1]),
                    lon=slice(lon[0], lon[-1]))

    print('Anomalía')
    for l in [0, 1, 2, 3]:
        try:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['r', 'time'])
        except:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['time'])

        if l == 0:
            anom = case.sel(time=case.time.dt.month.isin(mm - l)) #- clim_aux
        else:
            anom2 = case.sel(time=case.time.dt.month.isin(mm - l))# - clim_aux
            anom = xr.concat([anom, anom2], dim='time')

    print('1.2')
    anom = anom.sortby(anom.time.dt.month)

    if box is True:
        anom = anom.sel(lat=slice(min(box_lat), max(box_lat)),
                        lon=slice(box_lon[0], box_lon[1]))
        anom = anom.mean(['lon', 'lat'], skipna=True)
        aux_set_ds = ['time']
    else:
        aux_set_ds = ['time', 'lat', 'lon']

    # 2. Vinculo fechas case -> índices DMI y N34 para poder clasificarlos
    # las fechas entre el case variable y el case indices COINCIDEN,
    # DE ESA FORMA SE ELIGIERON LOS CASES VARIABLE
    # pero diferen en orden. Para evitar complicar la selección usando r y L
    # con .sortby(..time.dt.month) en cada caso se simplifica el problema
    # y coinciden todos los eventos en fecha, r y L

    cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates/'

    aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]: 'index'})

    case_sel_dmi = SelectVariables(aux_cases, data_dates_dmi_or)
    case_sel_dmi = case_sel_dmi.sortby(case_sel_dmi.time.dt.month)
    case_sel_dmi_n34 = SelectVariables(aux_cases, data_dates_n34_or)
    case_sel_dmi_n34 = case_sel_dmi_n34.sortby(case_sel_dmi_n34.time.dt.month)

    print('2.1 uniendo var, dmi y n34')
    try:
        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    # No deberia suceder pero con tref hay fechas duplicadas en 2011 y los campos
    # de estas fechas no son iguales. son 4 datos en total.
    except:
        print('error de anios, revisando...')
        times_to_remove = []
        for t in anom.time.values:
            lista = anom.sel(time=t).r.values
            vistos = set()

            try:
                len(lista)
                for valor in lista:
                    if valor in vistos:
                        print(f'error en anio {t}, será removido')
                        times_to_remove.append(t)
                    else:
                        vistos.add(valor)
            except:
                pass

        print(times_to_remove)
        for t in np.unique(times_to_remove):
            anom = anom.sel(time=anom.time != t)
            case_sel_dmi = case_sel_dmi.sel(
                time=case_sel_dmi.time != t)
            case_sel_dmi_n34 = case_sel_dmi_n34.sel(
                time=case_sel_dmi_n34.time != t)

        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    bins_aux_dmi = bins_by_cases_dmi[c_count]
    bins_aux_n34 = bins_by_cases_n34[c_count]
    print("3. Seleccion en cada bin")
    anom_bin_main = list()
    num_bin_main = list()
    # loops en las bins para el dmi segun case
    for ba_dmi in range(0, len(bins_aux_dmi)):
        bins_aux = data_merged.where(
            SelectBins(data_merged.dmi,
                       bin_limits[bins_aux_dmi[ba_dmi]][0],
                       bin_limits[bins_aux_dmi[ba_dmi]][1]))

        anom_bin = list()
        num_bin = list()
        # loop en las correspondientes al n34 segun case
        for ba_n34 in range(0, len(bins_aux_n34)):
            bin_f = bins_aux.where(
                SelectBins(
                    bins_aux.n34,
                    bin_limits[bins_aux_n34[ba_n34]][
                        0] / aux_n34_std.sst.values,
                    bin_limits[bins_aux_n34[ba_n34]][
                        1] / aux_n34_std.sst.values))

            if snr:
                spread = bin_f - bin_f.mean(['time'])
                spread = spread.std('time')
                SNR = bin_f.mean(['time']) / spread
                anom_bin.append(SNR)
            else:
                anom_bin.append(bin_f['var'])

            num_bin.append(len(np.where(~np.isnan(bin_f['dmi']))[0]))

        anom_bin_main.append(anom_bin)
        num_bin_main.append(num_bin)

    return anom_bin_main, num_bin_main, clim

# ---------------------------------------------------------------------------- #
def BinsByCases(v, v_name, fix_factor, s, mm, c, c_count,
                bin_limits, bins_by_cases_dmi, bins_by_cases_n34, dates_dir,
                cases_dir, snr=False, neutro_clim=False, box=False, box_lat=[],
                box_lon=[], ocean_mask=False):

    def Weights(data):
        weights = np.transpose(
            np.tile(np.cos(np.arange(
                data.lat.min().values, data.lat.max().values+1) * np.pi / 180),
                    (len(data.lon), 1)))
        try:
            data_w = data * weights
        except:
            data_w = data.transpose('time', 'lat', 'lon') * weights

        return data_w

    # 1. se abren los archivos de los índices (completos y se pesan por su SD)
    # tambien los archivos de la variable en cuestion pero para cada "case" = c

    data_dates_dmi_or = xr.open_dataset(dates_dir + 'DMI_' + s +
                                        '_Leads_r_CFSv2.nc')
    data_dates_dmi_or /=  data_dates_dmi_or.mean('r').std()

    data_dates_n34_or = xr.open_dataset(dates_dir + 'N34_' + s +
                                        '_Leads_r_CFSv2.nc')

    aux_n34_std =  data_dates_n34_or.mean('r').std()
    data_dates_n34_or /= aux_n34_std

    print('1.1 Climatología y case')
    if v != 'hgt':
        end_nc_file = '_detrend_05.nc'
        if v == 'hgt750': # que maravilla...
            end_nc_file = '__detrend_05.nc'
        #end_nc_file = '_no_detrend_05.nc'

    else:
        end_nc_file = '_05.nc'

    if neutro_clim:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_neutros' + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)
    else:
        clim = Weights(
            xr.open_dataset(cases_dir + v + '_' + s.lower() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)



    if ocean_mask is True:
        mask = MakeMask(clim, list(clim.data_vars)[0])
        clim = clim * mask

    try:
        case = Weights(
            xr.open_dataset(cases_dir + v + '_' + c + '_' + s.upper() +
                            end_nc_file).rename({v_name: 'var'}) * fix_factor)

        if ocean_mask is True:
            mask = MakeMask(case, list(case.data_vars)[0])
            case = case * mask

    except:
        print(f"case {c}, no encontrado para {v}")
        aux = clim.mean('time').__mul__(0)
        return aux, aux, aux

    try:
        clim = clim.sel(P=750)
        case = case.sel(P=750)
    except:
        pass

    if v == 'tref' or v == 'prec' or v == 'tsigma':
        lat = np.arange(-60, 20 + 1)
        lon = np.arange(275, 330 + 1)
    else:
        lat = np.linspace(-80, 20, 101)
        lon = np.linspace(0, 359, 360)

    if clim.lat[0]>clim.lat[-1]:
        lat_clim = lat[::-1]
    else:
        lat_clim = lat
    clim = clim.sel(lat=slice(lat_clim[0], lat_clim[-1]),
                    lon=slice(lon[0], lon[-1]))

    if case.lat[0] > case.lat[-1]:
        lat_case = lat[::-1]
    else:
        lat_case = lat
    case = case.sel(lat=slice(lat_case[0], lat_case[-1]),
                    lon=slice(lon[0], lon[-1]))

    print('Anomalía')
    for l in [0, 1, 2, 3]:
        try:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['r', 'time'])
        except:
            clim_aux = clim.sel(
                time=clim.time.dt.month.isin(mm - l)).mean(['time'])

        if l==0:
            anom = case.sel(time=case.time.dt.month.isin(mm - l)) - clim_aux
        else:
            anom2 = case.sel(time=case.time.dt.month.isin(mm - l)) - clim_aux
            anom = xr.concat([anom, anom2], dim='time')

    print( '1.2')
    anom = anom.sortby(anom.time.dt.month)

    if box is True:
        anom = anom.sel(lat=slice(min(box_lat), max(box_lat)),
                        lon=slice(box_lon[0], box_lon[1]))
        anom = anom.mean(['lon', 'lat'], skipna=True)
        aux_set_ds = ['time']
    else:
        aux_set_ds = ['time', 'lat', 'lon']

    # 2. Vinculo fechas case -> índices DMI y N34 para poder clasificarlos
    # las fechas entre el case variable y el case indices COINCIDEN,
    # DE ESA FORMA SE ELIGIERON LOS CASES VARIABLE
    # pero diferen en orden. Para evitar complicar la selección usando r y L
    # con .sortby(..time.dt.month) en cada caso se simplifica el problema
    # y coinciden todos los eventos en fecha, r y L

    cases_date_dir = '/pikachu/datos/luciano.andrian/cases_dates/'

    aux_cases = xr.open_dataset(cases_date_dir + c + '_f_' + s + '_05.nc')
    aux_cases = aux_cases.rename({list(aux_cases.data_vars)[0]: 'index'})

    case_sel_dmi = SelectVariables(aux_cases, data_dates_dmi_or)
    case_sel_dmi = case_sel_dmi.sortby(case_sel_dmi.time.dt.month)
    case_sel_dmi_n34 = SelectVariables(aux_cases, data_dates_n34_or)
    case_sel_dmi_n34 = case_sel_dmi_n34.sortby(case_sel_dmi_n34.time.dt.month)

    print('2.1 uniendo var, dmi y n34')
    try:
        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    # No deberia suceder pero con tref hay fechas duplicadas en 2011 y los campos
    # de estas fechas no son iguales. son 4 datos en total.
    except:
        print('error de anios, revisando...')
        times_to_remove = []
        for t in anom.time.values:
            lista = anom.sel(time=t).r.values
            vistos = set()

            try:
                len(lista)
                for valor in lista:
                    if valor in vistos:
                        print(f'error en anio {t}, será removido')
                        times_to_remove.append(t)
                    else:
                        vistos.add(valor)
            except:
                pass

        print(times_to_remove)
        for t in np.unique(times_to_remove):
            anom = anom.sel(time=anom.time != t)
            case_sel_dmi = case_sel_dmi.sel(
                time=case_sel_dmi.time != t)
            case_sel_dmi_n34 = case_sel_dmi_n34.sel(
                time=case_sel_dmi_n34.time != t)

        data_merged = xr.Dataset(
            data_vars=dict(
                var=(aux_set_ds, anom['var'].values),
                dmi=(['time'], case_sel_dmi.sst.values),
                n34=(['time'], case_sel_dmi_n34.sst.values),
            ),
            coords=dict(
                time=anom.time.values
            )
        )

    bins_aux_dmi = bins_by_cases_dmi[c_count]
    bins_aux_n34 = bins_by_cases_n34[c_count]
    print("3. Seleccion en cada bin")
    anom_bin_main = list()
    num_bin_main = list()
    # loops en las bins para el dmi segun case
    for ba_dmi in range(0, len(bins_aux_dmi)):
        bins_aux = data_merged.where(
            SelectBins(data_merged.dmi,
                       bin_limits[bins_aux_dmi[ba_dmi]][0],
                       bin_limits[bins_aux_dmi[ba_dmi]][1]))

        anom_bin = list()
        num_bin = list()
        # loop en las correspondientes al n34 segun case
        for ba_n34 in range(0, len(bins_aux_n34)):
            bin_f = bins_aux.where(
                SelectBins(
                    bins_aux.n34,
                    bin_limits[bins_aux_n34[ba_n34]][0]/aux_n34_std.sst.values,
                    bin_limits[bins_aux_n34[ba_n34]][1]/aux_n34_std.sst.values))

            if snr:
                spread = bin_f - bin_f.mean(['time'])
                spread = spread.std('time')
                SNR = bin_f.mean(['time']) / spread
                anom_bin.append(SNR)
            else:
                anom_bin.append(bin_f.mean('time')['var'])

            num_bin.append(len(np.where(~np.isnan(bin_f['dmi']))[0]))

        anom_bin_main.append(anom_bin)
        num_bin_main.append(num_bin)

    return anom_bin_main, num_bin_main, clim

# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #