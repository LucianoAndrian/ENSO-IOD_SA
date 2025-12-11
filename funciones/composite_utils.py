import pandas as pd
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------- #
def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)

# ---------------------------------------------------------------------------- #
def SelectYears(df, name_var, main_month=1, full_season=False):

    if full_season:
        print('Full Season JJASON')
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([7, 8, 9, 10, 11]))[name_var],
                            'Años': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Años'],
                            'Mes': df.where(df.Mes.isin([7, 8, 9, 10, 11]))['Mes']})
        mmin, mmax = 6, 11

    else:
        aux = pd.DataFrame({'Ind': df.where(df.Mes.isin([main_month]))[name_var],
                            'Años': df.where(df.Mes.isin([main_month]))['Años'],
                            'Mes': df.where(df.Mes.isin([main_month]))['Mes']})
        mmin, mmax = main_month - 1, main_month + 1

        if main_month == 1:
            mmin, mmax = 12, 2
        elif main_month == 12:
            mmin, mmax = 11, 1

    return aux.dropna(), mmin, mmax

# ---------------------------------------------------------------------------- #
def ClassifierEvents(df, full_season=False):
    if full_season:
        print('full season')
        df_pos = set(df.Años.values[np.where(df['Ind'] > 0)])
        df_neg = set(df.Años.values[np.where(df['Ind'] < 0)])
    else:
        df_pos = df.Años.values[np.where(df['Ind'] > 0)]
        df_neg = df.Años.values[np.where(df['Ind'] < 0)]

    return df_pos, df_neg

# ---------------------------------------------------------------------------- #
def NeutralEvents(df, mmin, mmax, start=1920, end = 2020, double=False,
                  df2=None, var_original=None):

    x = np.arange(start, end + 1, 1)

    start = str(start)
    end = str(end)

    mask = np.in1d(x, df.Años.values, invert=True)
    if mmax ==1: #NDJ
        print("NDJ Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]+1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=11, mmax=12))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(1))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    elif mmin == 12: #DJF
        print("DJF Special")
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        neutro_1 = var_original.sel(time=var_original.time.dt.year.isin(x[mask]-1))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_1 = neutro_1.sel(time=neutro_1.time.dt.year.isin(x[mask]))

        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=1, mmax=2))
        neutro_1 = neutro_1.sel(time=neutro_1.time.dt.month.isin(12))
        neutro = xr.merge([neutro, neutro_1])
        neutro = neutro.mean(['time'], skipna=True)

    else:
        mask = np.in1d(x, df.Años.values, invert=True)
        neutro = var_original.sel(time=var_original.time.dt.year.isin(x[mask]))
        if double:
            mask = np.in1d(x, df2.Años.values, invert=True)
            neutro = neutro.sel(time=neutro.time.dt.year.isin(x[mask]))
            neutro_years = list(set(neutro.time.dt.year.values))
        neutro = neutro.sel(time=is_months(month=neutro['time.month'], mmin=mmin, mmax=mmax))
        neutro = neutro.mean(['time'], skipna=True)

    return neutro, neutro_years

# ---------------------------------------------------------------------------- #
def Composite(original_data, index_pos, index_neg, mmin, mmax):
    comp_field_pos=0
    comp_field_neg=0

    if len(index_pos) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos+1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=11, mmax=12))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(1))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin(index_pos))
            comp_field_pos_1 = original_data.sel(time=original_data.time.dt.year.isin(index_pos - 1))

            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=1, mmax=2))
            comp_field_pos_1 = comp_field_pos_1.sel(time=comp_field_pos_1.time.dt.month.isin(2))

            comp_field_pos = xr.merge([comp_field_pos, comp_field_pos_1])
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])

        else:
            comp_field_pos = original_data.sel(time=original_data.time.dt.year.isin([index_pos]))
            comp_field_pos = comp_field_pos.sel(
                time=is_months(month=comp_field_pos['time.month'], mmin=mmin, mmax=mmax))
            if len(comp_field_pos.time) != 0:
                comp_field_pos = comp_field_pos.mean(['time'], skipna=True)
            else:
                comp_field_pos = comp_field_pos.drop_dims(['time'])


    if len(index_neg) != 0:
        if mmax == 1:
            print('NDJ Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg + 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=11, mmax=12))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(1))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if (len(comp_field_neg.time) != 0):
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dmis(['time'])

        elif mmin == 12:
            print('DJF Special')
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin(index_neg))
            comp_field_neg_1 = original_data.sel(time=original_data.time.dt.year.isin(index_neg - 1))

            comp_field_neg = comp_field_neg.sel(
                time=is_months(month=comp_field_neg['time.month'], mmin=1, mmax=2))
            comp_field_neg_1 = comp_field_neg_1.sel(time=comp_field_neg_1.time.dt.month.isin(2))

            comp_field_neg = xr.merge([comp_field_neg, comp_field_neg_1])
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

        else:
            comp_field_neg = original_data.sel(time=original_data.time.dt.year.isin([index_neg]))
            comp_field_neg = comp_field_neg.sel(time=is_months(month=comp_field_neg['time.month'],
                                                               mmin=mmin, mmax=mmax))
            if len(comp_field_neg.time) != 0:
                comp_field_neg = comp_field_neg.mean(['time'], skipna=True)
            else:
                comp_field_neg = comp_field_neg.drop_dims(['time'])

    return comp_field_pos, comp_field_neg

# ---------------------------------------------------------------------------- #
def MultipleComposite(var, n34, dmi, season,start = 1920, full_season=False,
                      compute_composite=False):

    seasons = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
               'JJA','JAS', 'ASO', 'SON', 'OND', 'NDJ']

    def check(x):
        if x is None:
            x = [0]
            return x
        else:
            if len(x) == 0:
                x = [0]
                return x
        return x

    if full_season:
        main_month_name = 'JJASON'
        main_month = None
    else:
        main_month, main_month_name = len(seasons[:season]) + 1, seasons[season]

    print(main_month_name)

    N34, N34_mmin, N34_mmax = SelectYears(df=n34, name_var='N34',
                                          main_month=main_month, full_season=full_season)
    DMI, DMI_mmin, DMI_mmax = SelectYears(df=dmi, name_var='DMI',
                                          main_month=main_month, full_season=full_season)
    DMI_sim_pos = [0,0]
    DMI_sim_neg = [0,0]
    DMI_un_pos = [0,0]
    DMI_un_neg = [0,0]
    DMI_pos = [0,0]
    DMI_neg = [0,0]
    N34_sim_pos = [0,0]
    N34_sim_neg = [0,0]
    N34_un_pos = [0,0]
    N34_un_neg = [0,0]
    N34_pos = [0,0]
    N34_neg = [0,0]
    All_neutral = [0, 0]

    if (len(DMI) != 0) & (len(N34) != 0):
        # All events
        DMI_pos, DMI_neg = ClassifierEvents(DMI, full_season=full_season)
        N34_pos, N34_neg = ClassifierEvents(N34, full_season=full_season)

        # both neutral, DMI and N34
        if compute_composite:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[0]

        else:
            All_neutral = NeutralEvents(df=DMI, mmin=DMI_mmin, mmax=DMI_mmax, start=start,
                                        df2=N34, double=True, var_original=var)[1]


        # Simultaneous events
        sim_events = np.intersect1d(N34.Años.values, DMI.Años.values)

        try:
            # Simultaneos events
            DMI_sim = DMI.where(DMI.Años.isin(sim_events)).dropna()
            N34_sim = N34.where(N34.Años.isin(sim_events)).dropna()
            DMI_sim_pos_aux, DMI_sim_neg_aux = ClassifierEvents(DMI_sim)
            N34_sim_pos_aux, N34_sim_neg_aux = ClassifierEvents(N34_sim)


            # Existen eventos simultaneos de signo opuesto?
            # cuales?
            sim_pos = np.intersect1d(DMI_sim_pos_aux, N34_sim_pos_aux)
            sim_pos2 = np.intersect1d(sim_pos, DMI_sim_pos_aux)
            DMI_sim_pos = sim_pos2

            sim_neg = np.intersect1d(DMI_sim_neg_aux, N34_sim_neg_aux)
            sim_neg2 = np.intersect1d(DMI_sim_neg_aux, sim_neg)
            DMI_sim_neg = sim_neg2


            if (len(sim_events) != (len(sim_pos) + len(sim_neg))):
                dmi_pos_n34_neg = np.intersect1d(DMI_sim_pos_aux, N34_sim_neg_aux)
                dmi_neg_n34_pos = np.intersect1d(DMI_sim_neg_aux, N34_sim_pos_aux)
            else:
                dmi_pos_n34_neg = None
                dmi_neg_n34_pos = None

            # Unique events
            DMI_un = DMI.where(-DMI.Años.isin(sim_events)).dropna()
            N34_un = N34.where(-N34.Años.isin(sim_events)).dropna()

            DMI_un_pos, DMI_un_neg = ClassifierEvents(DMI_un)
            N34_un_pos, N34_un_neg = ClassifierEvents(N34_un)

            if compute_composite:
                print('Making composites...')
                # ------------------------------------ SIMULTANEUS ---------------------------------------------#
                DMI_sim = Composite(original_data=var, index_pos=DMI_sim_pos, index_neg=DMI_sim_neg,
                                    mmin=DMI_mmin, mmax=DMI_mmax)

                # ------------------------------------ UNIQUES -------------------------------------------------#
                DMI_un = Composite(original_data=var, index_pos=DMI_un_pos, index_neg=DMI_un_neg,
                                   mmin=DMI_mmin, mmax=DMI_mmax)

                N34_un = Composite(original_data=var, index_pos=N34_un_pos, index_neg=N34_un_neg,
                                   mmin=N34_mmin, mmax=N34_mmax)
            else:
                print('Only dates, no composites')
                DMI_sim = None
                DMI_un = None
                N34_un = None

        except:
            DMI_sim = None
            DMI_un = None
            N34_un = None
            DMI_sim_pos = None
            DMI_sim_neg = None
            DMI_un_pos = None
            DMI_un_neg = None
            print('Only uniques events[3][4]')

        if compute_composite:
            # ------------------------------------ ALL ---------------------------------------------#
            dmi_comp = Composite(original_data=var, index_pos=list(DMI_pos), index_neg=list(DMI_neg),
                                 mmin=DMI_mmin, mmax=DMI_mmax)
            N34_comp = Composite(original_data=var, index_pos=list(N34_pos), index_neg=list(N34_neg),
                                 mmin=N34_mmin, mmax=N34_mmax)
        else:
            dmi_comp=None
            N34_comp=None

    DMI_sim_pos = check(DMI_sim_pos)
    DMI_sim_neg = check(DMI_sim_neg)
    DMI_un_pos = check(DMI_un_pos)
    DMI_un_neg = check(DMI_un_neg)
    DMI_pos = check(DMI_pos)
    DMI_neg = check(DMI_neg)

    N34_sim_pos = check(N34_sim_pos)
    N34_sim_neg = check(N34_sim_neg)
    N34_un_pos = check(N34_un_pos)
    N34_un_neg = check(N34_un_neg)
    N34_pos = check(N34_pos)
    N34_neg = check(N34_neg)

    DMI_pos_N34_neg = check(dmi_pos_n34_neg)
    DMI_neg_N34_pos = check(dmi_neg_n34_pos)

    All_neutral = check(All_neutral)


    if compute_composite:
        print('test')
        return DMI_sim, DMI_un, N34_un, dmi_comp, N34_comp, All_neutral, DMI_sim_pos, DMI_sim_neg, \
               DMI_un_pos, DMI_un_neg, N34_un_pos, N34_un_neg, DMI_pos, DMI_neg, N34_pos, N34_neg
    else:
        return list(All_neutral),\
               list(set(DMI_sim_pos)), list(set(DMI_sim_neg)),\
               list(set(DMI_un_pos)), list(set(DMI_un_neg)),\
               list(set(N34_un_pos)), list(set(N34_un_neg)),\
               list(DMI_pos), list(DMI_neg), \
               list(N34_pos), list(N34_neg), \
               list(DMI_pos_N34_neg), list(DMI_neg_N34_pos)

# ---------------------------------------------------------------------------- #
def CompositeSimple(original_data, index, mmin, mmax):
    def is_months(month, mmin, mmax):
        return (month >= mmin) & (month <= mmax)

    if len(index) != 0:
        comp_field = original_data.sel(time=original_data.time.dt.year.isin([index]))
        comp_field = comp_field.sel(
            time=is_months(month=comp_field['time.month'], mmin=mmin, mmax=mmax))
        if len(comp_field.time) != 0:
            comp_field = comp_field.mean(['time'], skipna=True)
        else:  # si sólo hay un año
            comp_field = comp_field.drop_dims(['time'])

        return comp_field
    else:
        print(' len index = 0')

# ---------------------------------------------------------------------------- #
def CaseComp(data, s, mmonth, c, two_variables=False, data2=None,
             return_neutro_comp=False, nc_date_dir='None'):
    """
    Las fechas se toman del periodo 1920-2020 basados en el DMI y N34 con ERSSTv5
    Cuando se toman los periodos 1920-1949 y 1950_2020 las fechas que no pertencen
    se excluyen de los composites en CompositeSimple()
    """
    mmin = mmonth[0]
    mmax = mmonth[-1]

    aux = xr.open_dataset(nc_date_dir + '1920_2020' + '_' + s + '.nc')
    neutro = aux.Neutral

    try:
        case = aux[c]
        case = case.where(case >= 1940)
        aux.close()

        case_num = len(case.values[np.where(~np.isnan(case.values))])
        case_num2 = case.values[np.where(~np.isnan(case.values))]

        neutro_comp = CompositeSimple(original_data=data, index=neutro, mmin=mmin, mmax=mmax)
        data_comp = CompositeSimple(original_data=data, index=case, mmin=mmin, mmax=mmax)

        comp = data_comp - neutro_comp

        if two_variables:
            neutro_comp2 = CompositeSimple(original_data=data2, index=neutro, mmin=mmin, mmax=mmax)
            data_comp2 = CompositeSimple(original_data=data2, index=case, mmin=mmin, mmax=mmax)

            comp2 = data_comp2 - neutro_comp2
        else:
            comp2 = None
    except:
        print('Error en ' + s + c)

    if two_variables:
        if return_neutro_comp:
            return comp, case_num, comp2, neutro_comp, neutro_comp2
        else:
            return comp, case_num, comp2
    else:
        if return_neutro_comp:
            return comp, case_num, neutro_comp
        else:
            return comp, case_num

# ---------------------------------------------------------------------------- #