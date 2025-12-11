import xarray as xr

def SelectVariables(dates, data):
    data_select = []

    for t in dates.index:
        try:
            r_t = t.r.values
        except:
            r_t = dates.r[0].values
        L_t = t.L.values
        t_t = t.values

        try:
            t_t * 1
            t_t = t.time.values
        except:
            pass

        mask = ((data.L == L_t) &
                (data.r == r_t) &
                (data.time == t_t))

        selected = data.where(mask, drop=True)
        data_select.append(selected.isel(r=0))

    return xr.concat(data_select, dim='time')
