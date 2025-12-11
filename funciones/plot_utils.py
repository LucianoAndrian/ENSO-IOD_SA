
import xarray as xr
import numpy as np
from numpy import ma
import pandas as pd
pd.options.mode.chained_assignment = None

import cartopy.feature
from matplotlib.font_manager import FontProperties
import string
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm

from funciones.general_utils import MakeMask

# ---------------------------------------------------------------------------- #
def SetDataToPlotFinal(*args):
    data_arrays = []
    first = True
    for arg in args:
        if not isinstance(arg, xr.DataArray):
            try:
                arg = xr.DataArray(arg, dims=['lat', 'lon'])
            except:
                arg = arg.to_array()
                if 1 in arg.shape:
                    arg = arg.squeeze()
                arg = xr.DataArray(arg, dims=['lat', 'lon'])
        else:
            if 1 in arg.shape:
                arg = arg.squeeze()

        if first is False:
            if data_arrays[0].lon.values[-1] != arg.lon.values[-1]:
                arg = arg.interp(lon = data_arrays[0].lon.values,
                                 lat = data_arrays[0].lat.values)

        data_arrays.append(arg)
        first = False

    data = xr.concat(data_arrays, dim='plots')
    data = data.assign_coords(plots=range(data.shape[0]))

    return data

# ---------------------------------------------------------------------------- #
def PlotFinal(data, levels, cmap, titles, namefig, map, save, dpi, out_dir,
              data_ctn=None, levels_ctn=None, color_ctn=None,
              data_ctn2=None, levels_ctn2=None, color_ctn2=None,
              data_waf=None, wafx=None, wafy=None, waf_scale=None,
              waf_step=None, waf_label=None, sig_points=None, hatches=None,
              num_cols=None, high=2, width = 7.08661, step=2, cbar_pos = 'H',
              num_cases=False, num_cases_data=None, pdf=False, ocean_mask=False,
              data_ctn_no_ocean_mask=False, data_ctn2_no_ocean_mask=False,
              pcolormesh=False):

    # cantidad de filas necesarias
    if num_cols is None:
        num_cols = 2
    width = width
    plots = data.plots.values
    num_plots = len(plots)
    num_rows = np.ceil(num_plots / num_cols).astype(int)

    crs_latlon = ccrs.PlateCarree()

    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        step_lon = 60
        high = high
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        step_lon = 60
        high = high
    elif map.upper() == 'HS_EX':
        extent = [0, 359, -65, -20]
        step_lon = 60
        high = 2
    elif map.upper() == 'SA':
        extent = [275, 330, -60, 20]
        step_lon = 20
        high = high
    else:
        print(f"Mapa {map} no seteado")
        return

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    i = 0
    for ax, plot in zip(axes.flatten(), plots):
        no_plot = False

        # CONTOUR ------------------------------------------------------------ #
        if data_ctn is not None:
            if levels_ctn is None:
                levels_ctn = levels.copy()
            try:
                if isinstance(levels_ctn, np.ndarray):
                    levels_ctn = levels_ctn[levels_ctn != 0]
                else:
                    levels_ctn.remove(0)
            except:
                pass
            aux_ctn = data_ctn.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values

                ax.contour(data_ctn.lon.values[::step],
                           data_ctn.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.4,
                           levels=levels_ctn, transform=crs_latlon,
                           colors=color_ctn)

        # CONTOUR2 ----------------------------------------------------------- #
        if data_ctn2 is not None:
            if levels_ctn2 is None:
                levels_ctn2 = levels.copy()

            try:
                if isinstance(levels_ctn2, np.ndarray):
                    levels_ctn2 = levels_ctn2[levels_ctn != 0]
                else:
                    levels_ctn2.remove(0)
            except:
                pass
            aux_ctn = data_ctn2.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn2_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values
                ax.contour(data_ctn2.lon.values[::step],
                           data_ctn2.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.5,
                           levels=levels_ctn2, transform=crs_latlon,
                           colors=color_ctn2)

        # CONTOURF OR COLORMESH ---------------------------------------------- #
        aux = data.sel(plots=plot)
        if aux.mean().values != 0:

            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux = aux * mask_ocean.mask

            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            if pcolormesh is True:
                im = ax.pcolormesh(aux.lon.values[::step],
                                   aux.lat.values[::step],
                                   aux_var[::step, ::step],
                                   vmin=np.min(levels), vmax=np.max(levels),
                                   transform=crs_latlon, cmap=cmap)
            else:
                im = ax.contourf(aux.lon.values[::step], aux.lat.values[::step],
                                 aux_var[::step, ::step],
                                 levels=levels,
                                 transform=crs_latlon, cmap=cmap, extend='both')


        else:
            ax.axis('off')
            no_plot=True

        # WAF ---------------------------------------------------------------- #
        if data_waf is not None:
            wafx_aux = wafx.sel(plots=plot)
            wafy_aux = wafy.sel(plots=plot)

            if ocean_mask is True:
                mask_ocean = MakeMask(wafx_aux)
                wafx_aux = wafx_aux * mask_ocean.mask
                wafy_aux = wafy_aux * mask_ocean.mask


            Q60 = np.nanpercentile(np.sqrt(np.add(np.power(wafx_aux, 2),
                                                  np.power(wafy_aux, 2))), 60)
            M = np.sqrt(np.add(np.power(wafx_aux, 2),
                               np.power(wafy_aux, 2))) < Q60
            # mask array
            wafx_mask = ma.array(wafx_aux, mask=M)
            wafy_mask = ma.array(wafy_aux, mask=M)
            Q99 = np.nanpercentile(np.sqrt(np.add(np.power(wafx_aux, 2),
                                                  np.power(wafy_aux, 2))), 99)
            M = np.sqrt(np.add(np.power(wafx_aux, 2),
                               np.power(wafy_aux, 2))) > Q99
            # mask array
            wafx_mask = ma.array(wafx_mask, mask=M)
            wafy_mask = ma.array(wafy_mask, mask=M)

            # plot vectors
            lons, lats = np.meshgrid(data_waf.lon.values, data_waf.lat.values)
            Q = ax.quiver(lons[::waf_step, ::waf_step],
                          lats[::waf_step, ::waf_step],
                          wafx_mask[::waf_step, ::waf_step],
                          wafy_mask[::waf_step, ::waf_step],
                          transform=crs_latlon, pivot='tail',
                          width=1.7e-3, headwidth=4, alpha=1, headlength=2.5,
                          color='k', scale=waf_scale, angles='xy',
                          scale_units='xy')

            ax.quiverkey(Q, 0.85, 0.05, waf_label,
                         f'{waf_label:.1e} $m^2$ $s^{{-2}}$',
                         labelpos='E', coordinates='figure', labelsep=0.05,
                         fontproperties=FontProperties(size=6, weight='light'))

        # SIG ---------------------------------------------------------------- #
        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']
                try:
                    comp_sig_var = aux_sig_points['var']
                except:
                    comp_sig_var = aux_sig_points.values
                cs = ax.contourf(aux_sig_points.lon[::step],
                                 aux_sig_points.lat[::step],
                                 comp_sig_var[::step, ::step],
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower')

                for i2, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i2 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)

        # no plotear --------------------------------------------------------- #
        if no_plot is False:
            if num_cases:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]}), "
                                       f"$N={num_cases_data[plot]}$",
                        transform=ax.transAxes, size=6)
            else:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes, size=6)
            i = i + 1

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2,
                          resolution='110m')
            if map.upper() == 'SA':
                ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                               linestyle='-', linewidth=0.2, color='k')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                              zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            ax.set_xticks(np.arange(0, 360, step_lon), crs=crs_latlon)
            ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            ax.tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=4)
            ax.set_extent(extent, crs=crs_latlon)

            ax.set_aspect('equal')
            ax.set_title(titles[plot], fontsize=6, pad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    #cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.235, 0.03, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=5, pad=1)
        fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        aux_color = cmap.colors[2]
        patch = mpatches.Patch(color=aux_color, label='Ks < 0')

        legend = fig.legend(handles=[patch], loc='lower center', fontsize=8,
                            frameon=True, framealpha=1, fancybox=True)
        legend.set_bbox_to_anchor((0.5, 0.01), transform=fig.transFigure)
        legend.get_frame().set_linewidth(0.5)
        fig.subplots_adjust(bottom=0.1, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{namefig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
def PlotFinal_CompositeByMagnitude(data, levels, cmap, titles, namefig, map,
                                   save, dpi, out_dir, data_ctn=None,
                                   levels_ctn=None, color_ctn=None,
                                   row_titles=None, col_titles=None,
                                   clim_plot=None, clim_levels=None,
                                   clim_cbar=None, high=2, width = 7.08661,
                                   cbar_pos='H', plot_step=3,
                                   clim_plot_ctn=None, clim_levels_ctn=None,
                                   pdf=True, ocean_mask=False,
                                   data_ctn_no_ocean_mask=False,
                                   sig_data=None, hatches=None,
                                   plot_regiones=False):

    # cantidad de filas necesarias
    num_cols = 5
    num_rows = 5

    plots = data.plots.values
    crs_latlon = ccrs.PlateCarree()

    # mapa
    if map.upper() == 'HS':
        extent = [0, 359, -80, 20]
        high = high
        xticks = np.arange(0, 360, 60)
        yticks = np.arange(-80, 20, 20)
        lon_localator = 60
    elif map.upper() == 'TR':
        extent = [45, 270, -20, 20]
        high = high
        xticks = np.arange(0, 360, 60)
        np.arange(-80, 20, 20)
        lon_localator = 60
    elif map.upper() == 'HS_EX':
        extent = [0, 359, -65, -20]
        high = high
        xticks = np.arange(0, 360, 60)
        lon_localator = 60
    elif map.upper() == 'SA':
        extent = [270, 330, -60, 20]
        high = high
        yticks = np.arange(-60, 15+1, 20)
        xticks = np.arange(275, 330+1, 20)
        lon_localator = 20
    else:
        print(f"Mapa {map} no seteado")
        return

    # plot
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.05, 'hspace': 0.01})

    i2 = 0
    for i, (ax, plot) in enumerate(zip(axes.flatten(), plots)):
        remove_axes = False

        # Seteo filas, titulos ----------------------------------------------- #
        if i == 2 or i == 10:
            ax.set_title(f'{col_titles[i]}\n                           ',
                         fontsize=5, pad=2, loc='left')
            ax.yaxis.set_label_position('left')
            ax.text(-0.07, 0.5, row_titles[i], rotation=90,
                    transform=ax.transAxes, fontsize=5,
                    verticalalignment='center')
        elif i == 3 or i == 4 or i == 11:
            ax.set_title(f'{col_titles[i]}\n                           ',
                         fontsize=5, pad=3, loc='left')
        elif i == 7 or i == 15 or i == 20:
            ax.yaxis.set_label_position('left')
            ax.text(-0.07, 0.5, row_titles[i], rotation=90,
                    transform=ax.transAxes, fontsize=5,
                    verticalalignment='center')

        if i in [4, 9, 14, 17, 22]:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_yticks(yticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)
            # ax.tick_params(labelsize=3)
            # ax.set_extent(extent, crs=crs_latlon)

        if i in [20,21,22, 23, 13, 14]:
            ax.set_xticks(xticks, crs=crs_latlon)
            ax.tick_params(width=0.3, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.tick_params(labelsize=3)
            #ax.set_extent(extent, crs=crs_latlon)

        ax.tick_params(width=0.5, pad=1, labelsize=4)
        # Plot --------------------------------------------------------------- #
        if plot == 12:  # Clima
            if clim_plot.mean()['var'].values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(clim_plot)
                    clim_plot = clim_plot * mask_ocean.mask

                ax_new = fig.add_axes(
                    [0.365, 0.41, 0.19, 0.18],
                    projection=ccrs.PlateCarree())

                # Contour ---------------------------------------------------- #
                if clim_plot_ctn and clim_plot_ctn.mean()['var'].values != 0:

                    if ocean_mask is True and data_ctn_no_ocean_mask is False:
                        mask_ocean = MakeMask(clim_plot_ctn)
                        clim_plot_ctn = clim_plot_ctn * mask_ocean.mask

                    ax_new.contour(clim_plot_ctn.lon.values[::plot_step],
                                   clim_plot_ctn.lat.values[::plot_step],
                                   clim_plot_ctn['var'][::plot_step, ::plot_step],
                                   linewidths=0.4,
                                   levels=clim_levels_ctn, transform=crs_latlon,
                                   colors=color_ctn)

                # Contourf --------------------------------------------------- #
                cp = ax_new.contourf(clim_plot.lon.values[::plot_step],
                                     clim_plot.lat.values[::plot_step],
                                     clim_plot['var'][::plot_step,
                                     ::plot_step],
                                     levels=clim_levels,
                                     transform=crs_latlon,
                                     cmap=clim_cbar,
                                     extend='both')

                ax_new.set_title('Plot 12', fontsize=5)

                # Barra de colores si es necesario
                cb = plt.colorbar(cp, ax=ax_new, fraction=0.046,
                                  pad=0.02, shrink=0.6,
                                  orientation='horizontal')
                cb.ax.tick_params(labelsize=4, pad=0.1, length=1,
                                  width=0.5)
                for spine in cb.ax.spines.values():
                    spine.set_linewidth(0.5)

                else:
                    pass

                ax_new.add_feature(cartopy.feature.LAND, facecolor='white')
                ax_new.coastlines(color='k', linestyle='-', alpha=1,
                                  resolution='110m', linewidth=0.2)
                ax_new.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) ",
                        transform=ax_new.transAxes, size=4)
                i2 += 1
                ax_new.set_title('Climatology', fontsize=4, pad=1)
                gl = ax_new.gridlines(draw_labels=False, linewidth=0.3,
                                      linestyle='-', zorder=20)

                gl.ylocator = plt.MultipleLocator(20)
                gl.xlocator = plt.MultipleLocator(60)

                if plot_regiones is True:
                    names_regiones_sa = ['Am', 'NeB', 'N-SESA', 'S-SESA',
                                         'Andes-C', 'Andes-S']
                    lat_regiones_sa = [[-13, 2], [-15, 2], [-27, -15],
                                       [-39, -25], [-45, -30], [-56, -45]]
                    lon_regiones_sa = [[291, 304], [311, 326], [306, 325],
                                       [296, 306], [285, 293], [284, 290]]

                    for r, rname in enumerate(names_regiones_sa):
                        w = lon_regiones_sa[r][1] - lon_regiones_sa[r][0]
                        h = np.abs(lat_regiones_sa[r][0]) - np.abs(
                            lat_regiones_sa[r][1])
                        ax_new.add_patch(
                            mpatches.Rectangle(
                                xy=[lon_regiones_sa[r][0], lat_regiones_sa[r][0]],
                                width=w, height=h, facecolor='None', alpha=1,
                                edgecolor='k', linewidth=0.8,
                                transform=ccrs.PlateCarree()))

                for spine in ax_new.spines.values():
                    spine.set_linewidth(0.5)

            ax.axis('off')

        else:
            # Contour -------------------------------------------------------- #
            if data_ctn is not None:
                if levels_ctn is None:
                    levels_ctn = levels.copy()
                try:
                    if isinstance(levels_ctn, np.ndarray):
                        levels_ctn = levels_ctn[levels_ctn != 0]
                    else:
                        levels_ctn.remove(0)
                except:
                    pass

                aux_ctn = data_ctn.sel(plots=plot)
                if ((aux_ctn.mean().values != 0) and
                        (~np.isnan(aux_ctn.mean().values))):

                    if ocean_mask is True and data_ctn_no_ocean_mask is False:
                        mask_ocean = MakeMask(aux_ctn)
                        aux_ctn = aux_ctn * mask_ocean.mask

                    try:
                        aux_ctn_var = aux_ctn['var'].values
                    except:
                        aux_ctn_var = aux_ctn.values

                    ax.contour(data_ctn.lon.values[::plot_step],
                               data_ctn.lat.values[::plot_step],
                               aux_ctn_var[::plot_step,::plot_step],
                               linewidths=0.6,
                               levels=levels_ctn, transform=crs_latlon,
                               colors=color_ctn)
                else:
                    remove_axes = True

            aux = data.sel(plots=plot)
            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values


            if sig_data is not None:
                aux_sig_points = sig_data.sel(plots=plot)
                if aux_sig_points.mean().values != 0:

                    if ocean_mask is True:
                        mask_ocean = MakeMask(aux_sig_points)
                        aux_sig_points = aux_sig_points * mask_ocean.mask

                    # hatches = '....'
                    colors_l = ['k', 'k']
                    try:
                        comp_sig_var = aux_sig_points['var']
                    except:
                        comp_sig_var = aux_sig_points.values
                    cs = ax.contourf(aux_sig_points.lon,
                                     aux_sig_points.lat,
                                     comp_sig_var,
                                     transform=crs_latlon, colors='none',
                                     hatches=[hatches, hatches], extend='lower', zorder=5)

                    for i3, collection in enumerate(cs.collections):
                        collection.set_edgecolor(colors_l[i3 % len(colors_l)])

                    for collection in cs.collections:
                        collection.set_linewidth(0.)


            # Contourf ------------------------------------------------------- #
            if ((aux.mean().values != 0) and
                    (~np.isnan(aux.mean().values))):

                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i2]}) "
                                       f"$N={titles[plot]}$",
                        transform=ax.transAxes, size=4)

                i2 += 1

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux)
                    aux_var = aux_var * mask_ocean.mask

                im = ax.contourf(aux.lon.values[::plot_step],
                                 aux.lat.values[::plot_step],
                                 aux_var[::plot_step,::plot_step],
                                 levels=levels,
                                 transform=crs_latlon, cmap=cmap, extend='both', zorder=1)

                ax.add_feature(cartopy.feature.LAND, facecolor='white',
                               linewidth=0.5)
                ax.coastlines(color='k', linestyle='-', alpha=1, linewidth=0.2,
                              resolution='110m')
                gl = ax.gridlines(draw_labels=False, linewidth=0.1,
                                  linestyle='-', zorder=20)
                gl.ylocator = plt.MultipleLocator(20)
                gl.xlocator = plt.MultipleLocator(lon_localator)

            else:
                remove_axes = True

            if remove_axes:
                ax.axis('off')

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    # cbar_pos = 'H'
    if cbar_pos.upper() == 'H':
        pos = fig.add_axes([0.261, 0, 0.5, 0.02])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='horizontal')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0.05, wspace=0, hspace=0.25, left=0, right=1,
                            top=1)

    elif cbar_pos.upper() == 'V':
        pos = fig.add_axes([0.95, 0.2, 0.02, 0.5])
        cb = fig.colorbar(im, cax=pos, pad=0.1, orientation='vertical')
        cb.ax.tick_params(labelsize=4, pad=1)
        fig.subplots_adjust(bottom=0, wspace=0.5, hspace=0.25, left=0.02,
                            right=0.9, top=1)
    else:
        print(f"cbar_pos {cbar_pos} no valido")

    if save:
        if pdf:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        else:
            plt.savefig(f"{out_dir}{namefig}.jpg", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
def PlotPdfs(data, selected_cases, width=5, high=1.2, title='', namefig='fig',
             out_dir='', save=False, dpi=100):

    positive_cases_colors = ['red', '#F47B00', 'forestgreen']
    negative_cases_colors = ['#509DFE', '#00E071', '#FF3AA0']
    colors_cases = [positive_cases_colors, negative_cases_colors]

    fig, axes = plt.subplots(
        1, 2, figsize=(width, high * 2),
        gridspec_kw={'wspace': 0.05, 'hspace': 0.01})

    for ax_count, (ax, cases, color_case) in enumerate(zip(
            axes.flatten(), selected_cases, colors_cases)):
        if ax_count == 1:
            ax.yaxis.tick_right()

        max_y = []
        ax.plot(data['clim'], lw=2.5, color='k', label='Clim.')
        max_y.append(max(data['clim']))
        for c_count, c in enumerate(cases[1::]):
            aux_case = data[c]
            ax.plot(aux_case, lw=2.5, color=color_case[c_count], label=c)
            max_y.append(max(aux_case))

        ax.grid(alpha=0.5)
        ax.legend(loc='best')
        ax.set_ylim(0, max(max_y) + 0.001)

    fig.suptitle(title, fontsize=10)
    plt.yticks(size=10)
    plt.xticks(size=10)
    plt.tight_layout()
    if save:
        plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
def PlotBars(x, bin_n, bin_n_err, bin_n_len,
             bin_d, bin_d_err, bin_d_len,
             title='', name_fig='fig', out_dir='out_dir', save=False,
             ymin=-80, ymax=45, dpi=100, ylabel='Anomaly',
             bar_n_color=None, bar_n_error_color=None, bar_d_color=None,
             bar_d_error_color=None):

    fig = plt.figure(1, figsize=(7, 7), dpi=dpi)
    ax = fig.add_subplot(111)

    plt.hlines(y=0, xmin=-4, xmax=4, color='k')

    ax.bar(x + 0.075, bin_n, color=bar_n_color, alpha=1, width=0.15,
           label='Niño3.4')
    ax.errorbar(x + 0.075, bin_n, yerr=bin_n_err, capsize=4, fmt='o', alpha=1,
                elinewidth=0.9, ecolor=bar_n_error_color, mfc='w',
                mec=bar_n_error_color, markersize=5)

    ax2 = ax.twinx()
    ax2.bar(x + 0.075, bin_n_len, color=bar_n_color, alpha=0.7, width=0.15)

    ax.bar(x - 0.075, np.nan_to_num(bin_d), color=bar_d_color, alpha=1,
           width=0.15, label='DMI')
    ax.errorbar(x - 0.075, bin_d, yerr=bin_d_err, capsize=4, fmt='o', alpha=1,
                elinewidth=0.9, ecolor=bar_d_error_color, mec=bar_d_error_color,
                mfc='w', markersize=5)
    ax2.bar(x - 0.075, bin_d_len, color=bar_d_color, alpha=0.7, width=0.15)

    ax.legend(loc='upper left')
    ax.set_ylim(ymin, ymax)
    ax2.set_ylim(0, 3000)
    ax.set_ylabel(ylabel, fontsize=10)
    ax2.set_ylabel('number of samples', fontsize=10)
    ax.set_xlabel('SST index (of std)', fontsize=15)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.grid(True)
    plt.title(title, fontsize=15)
    plt.xlim(-3.5, 3.5)

    if save:
        plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
def PlotPDFTable(df, cmap, levels, title, name_fig='fig',
                 save=False, out_dir='~/', dpi=100, color_thr=0.4):

    fig = plt.figure(dpi=dpi, figsize=(8, 4))
    ax = fig.add_subplot(111)
    norm = BoundaryNorm(levels, cmap.N, clip=True)

    im = ax.imshow(df, cmap=cmap, norm=norm, aspect='auto')

    data_array = df.values
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            if np.abs(data_array[i, j]) > color_thr:
                color_num = 'white'
            else:
                color_num = 'k'
            ax.text(j, i, f"{data_array[i, j]:.2f}", ha='center', va='center',
                    color=color_num)

    # Ticks principales en el centro de las celdas (para los labels)
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=0, ha='center')

    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)

    ax.set_xticks(np.arange(-0.5, df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df.shape[0], 1), minor=True)

    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)

    fig.suptitle(title, size=12)

    ax.margins(0)
    plt.tight_layout()

    if save:
        plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
def PlotFinalTwoVariables(data, num_cols,
                          levels_r1, cmap_r1,
                          levels_r2, cmap_r2,
                          data_ctn, levels_ctn_r1, levels_ctn_r2, color_ctn,
                          titles, namefig, save, dpi, out_dir, pdf=False,
                          high=2, width = 7.08661, step=1,
                          ocean_mask=False, num_cases=False,
                          num_cases_data=None,
                          sig_points=None, hatches=None,
                          data_ctn_no_ocean_mask=False):

    plots = data.plots.values
    num_plots = len(plots)
    num_rows = np.ceil(num_plots / num_cols).astype(int)

    crs_latlon = ccrs.PlateCarree()

    map == 'SA'
    extent = [275, 330, -60, 20]
    step_lon = 20
    high = high

    change_row = len(data.plots)/2

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    i = 0
    for ax, plot in zip(axes.flatten(), plots):
        no_plot = False

        if plot < change_row:
            levels = levels_r1
            levels_ctn = levels_ctn_r1
            cmap = cmap_r1
            row1=True
        else:
            levels = levels_r2
            levels_ctn = levels_ctn_r2
            cmap = cmap_r2
            row1 = False


        # CONTOUR ------------------------------------------------------------ #
        if data_ctn is not None:
            if levels_ctn is None:
                levels_ctn = levels.copy()
            try:
                if isinstance(levels_ctn, np.ndarray):
                    levels_ctn = levels_ctn[levels_ctn != 0]
                else:
                    levels_ctn.remove(0)
            except:
                pass
            aux_ctn = data_ctn.sel(plots=plot)
            if aux_ctn.mean().values != 0:

                if ocean_mask is True and data_ctn_no_ocean_mask is False:
                    mask_ocean = MakeMask(aux_ctn)
                    aux_ctn = aux_ctn * mask_ocean.mask

                try:
                    aux_ctn_var = aux_ctn['var'].values
                except:
                    aux_ctn_var = aux_ctn.values

                ax.contour(data_ctn.lon.values[::step],
                           data_ctn.lat.values[::step],
                           aux_ctn_var[::step, ::step], linewidths=0.6,
                           levels=levels_ctn, transform=crs_latlon,
                           colors=color_ctn)

        aux = data.sel(plots=plot)
        if aux.mean().values != 0:

            if ocean_mask is True:
                mask_ocean = MakeMask(aux)
                aux = aux * mask_ocean.mask

            try:
                aux_var = aux['var'].values
            except:
                aux_var = aux.values

            if row1 is True:
                im_r1= ax.contourf(aux.lon.values[::step],
                                   aux.lat.values[::step],
                                   aux_var[::step, ::step],
                                   levels=levels,
                                   transform=crs_latlon, cmap=cmap,
                                   extend='both')
            else:
                im_r2 = ax.contourf(aux.lon.values[::step],
                                    aux.lat.values[::step],
                                    aux_var[::step, ::step],
                                    levels=levels,
                                    transform=crs_latlon, cmap=cmap,
                                    extend='both')
        else:
            ax.axis('off')
            no_plot=True


        # sig ---------------------------------------------------------------- #
        if sig_points is not None:
            aux_sig_points = sig_points.sel(plots=plot)
            if aux_sig_points.mean().values != 0:

                if ocean_mask is True:
                    mask_ocean = MakeMask(aux_sig_points)
                    aux_sig_points = aux_sig_points * mask_ocean.mask

                # hatches = '....'
                colors_l = ['k', 'k']
                try:
                    comp_sig_var = aux_sig_points['var']
                except:
                    comp_sig_var = aux_sig_points.values
                cs = ax.contourf(aux_sig_points.lon[::step],
                                 aux_sig_points.lat[::step],
                                 comp_sig_var[::step, ::step],
                                 transform=crs_latlon, colors='none',
                                 hatches=[hatches, hatches], extend='lower')

                for i2, collection in enumerate(cs.collections):
                    collection.set_edgecolor(colors_l[i2 % len(colors_l)])

                for collection in cs.collections:
                    collection.set_linewidth(0.)


        # no plotear --------------------------------------------------------- #
        if no_plot is False:
            if num_cases:
                ax.text(-0.01, 1.055, f"({string.ascii_lowercase[i]}) "
                                       f"$N={num_cases_data[plot]}$",
                        transform=ax.transAxes, size=6)
            else:
                ax.text(-0.005, 1.025, f"({string.ascii_lowercase[i]})",
                        transform=ax.transAxes, size=6)
            i = i + 1

            ax.add_feature(cartopy.feature.LAND, facecolor='white',
                           linewidth=0.5)
            ax.coastlines(color='k', linestyle='-', alpha=1,
                          linewidth=0.2,
                          resolution='110m')

            ax.add_feature(cartopy.feature.BORDERS, alpha=1,
                               linestyle='-', linewidth=0.2, color='k')
            gl = ax.gridlines(draw_labels=False, linewidth=0.1, linestyle='-',
                              zorder=20)
            gl.ylocator = plt.MultipleLocator(20)
            ax.set_xticks(np.arange(0, 360, step_lon), crs=crs_latlon)
            ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
            ax.tick_params(width=0.5, pad=1)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            ax.tick_params(labelsize=4)
            ax.set_extent(extent, crs=crs_latlon)

            ax.set_aspect('equal')
            ax.set_title(titles[plot], fontsize=6, pad=2)

            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    pos1 = fig.add_axes([0.87, 0.55, 0.015, 0.35])
    cb1 = fig.colorbar(im_r1, cax=pos1, orientation='vertical')
    cb1.ax.tick_params(labelsize=5, pad=1)

    pos2 = fig.add_axes([0.87, 0.1, 0.015, 0.35])
    cb2 = fig.colorbar(im_r2, cax=pos2, orientation='vertical')
    cb2.ax.tick_params(labelsize=5, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0.05, hspace=0.25, left=0.05,
                        right=0.85, top=0.95)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{namefig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{namefig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()


    plt.show()

# ---------------------------------------------------------------------------- #
def PlotBins2DTwoVariables(data_bins, num_bins, bin_limits, num_cols,
                           variable_v1, variable_v2, levels_v1, cmap_v1,
                           levels_v2, cmap_v2, color_thr_v1, color_thr_v2,
                           title, save, name_fig, out_dir, dpi,
                           high=3.5, width=11, pdf=True):

    num_plots = len(data_bins)
    num_rows = np.ceil(num_plots / num_cols).astype(int)
    change_row = len(data_bins)/2

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(width, high * num_rows),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.2})

    for ax, plot in zip(axes.flatten(), range(0, num_plots)):

        if plot < change_row:
            levels = levels_v1
            cmap = cmap_v1
            color_thr = color_thr_v1
            row1=True
            ylabel = variable_v1
        else:
            levels = levels_v2
            cmap = cmap_v2
            color_thr = color_thr_v2
            row1 = False
            ylabel = variable_v2

        data_sel = data_bins[plot]
        if data_sel is None:
            to_plot = False
        else:
            to_plot = True

        num_bins_sel = num_bins[plot]
        cmap = plt.get_cmap(cmap)

        norm = BoundaryNorm(levels, cmap.N, clip=True)

        if to_plot is True:
            if row1 is True:
                im_r1 = ax.imshow(data_sel, cmap=cmap, norm=norm)
            else:
                im_r2 = ax.imshow(data_sel, cmap=cmap, norm=norm)

            for i in range(0, len(bin_limits)):
                for j in range(0, len(bin_limits)):
                    if np.abs(data_sel[i, j]) > color_thr:
                        color_num = 'white'
                    else:
                        color_num = 'k'
                    if (~np.isnan(num_bins_sel[i, j])) and (
                            num_bins_sel[i, j] != 0):
                        ax.text(j, i, num_bins_sel[i, j].astype(np.int64),
                                ha='center', va='center', color=color_num,
                                size=9)

            ax.set_title(title[plot])
            ax.text(-0.01, 1.025, f"({string.ascii_lowercase[plot]})",
                    transform=ax.transAxes, size=9)

            xylimits = [-.5, -.5 + len(bin_limits)]
            ax.set_xlim(xylimits[::-1])
            ax.set_ylim(xylimits)

            original_ticks = np.arange(-.5, -.5 + len(bin_limits) + 0.5)
            new_tickx = np.unique(bin_limits)
            ax.set_xticks(original_ticks, new_tickx)
            ax.set_yticks(original_ticks, new_tickx)
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=8)

            inf_neutro_border = original_ticks[
                                    int(np.floor(len(original_ticks) / 2))] - 1
            upp_neutro_border = original_ticks[
                int(np.ceil(len(original_ticks) / 2))]
            ax.axhline(y=inf_neutro_border, color='k', linestyle='-',
                       linewidth=2)
            ax.axhline(y=upp_neutro_border, color='k', linestyle='-',
                       linewidth=2)
            ax.axvline(x=inf_neutro_border, color='k', linestyle='-',
                       linewidth=2)
            ax.axvline(x=upp_neutro_border, color='k', linestyle='-',
                       linewidth=2)

            ax.margins(0)
            ax.grid(which='major', alpha=0.5, color='k')
            if plot == 0 or plot == change_row:
                ax.set_ylabel(ylabel)
                first = False
        else:
            ax.axis('off')


    pos1 = fig.add_axes([0.93, 0.57, 0.015, 0.35])
    cb1 = fig.colorbar(im_r1, cax=pos1, orientation='vertical')
    cb1.ax.tick_params(labelsize=8, pad=1)

    pos2 = fig.add_axes([0.93, 0.1, 0.015, 0.35])
    cb2 = fig.colorbar(im_r2, cax=pos2, orientation='vertical')
    cb2.ax.tick_params(labelsize=8, pad=1)

    fig.subplots_adjust(bottom=0.1, wspace=0.05, hspace=0.55, left=0.075,
                        right=0.90, top=0.95)

    fig.supylabel('Niño3.4 - SST index (of std)', fontsize=11)
    fig.supxlabel('DMI - SST index (of std)', fontsize=11)

    if save:
        if pdf is True:
            plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.savefig(f"{out_dir}{name_fig}.png", dpi=dpi, bbox_inches='tight')
            plt.close()
    else:
        plt.show()

# ---------------------------------------------------------------------------- #
