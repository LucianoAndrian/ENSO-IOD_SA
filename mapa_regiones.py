
"""
Mapa de regiones
"""
################################################################################
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'
save = False
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
import matplotlib.patches as mpatches
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore")
################################################################################
if save:
    dpi = 300
else:
    dpi = 100
################################################################################
names_regiones_sa = ['Am', 'NeB', 'N-SESA', 'S-SESA', 'Chile-Cuyo', 'Patagonia']#, 'Aux_20']
lat_regiones_sa = [[-13, 2], [-15, 2], [-29, -17], [-39, -25], [-40,-30], [-56, -40]]#, [-25, -15]]
lon_regiones_sa = [[291, 304], [311, 325], [303, 315], [296, 306], [285,293], [287, 295]]#, [305, 320]]

################################################################################
# Plot regiones SA ------------------------------------------------------------#
print('plot regiones')
fig = plt.figure(figsize=(3, 4), dpi=dpi)
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
crs_latlon = ccrs.PlateCarree()
extent = [275, 330, -60, 20]


for r, rname in enumerate(names_regiones_sa):
    w = lon_regiones_sa[r][1] - lon_regiones_sa[r][0]
    h = np.abs(lat_regiones_sa[r][0]) - np.abs(lat_regiones_sa[r][1])
    ax.add_patch(
        mpatches.Rectangle(xy=[lon_regiones_sa[r][0],lat_regiones_sa[r][0]],
                           width = w, height=h, facecolor='None', alpha=1,
                           edgecolor='k', linewidth=2,
                           transform=ccrs.PlateCarree()))

ax.add_feature(cartopy.feature.LAND, facecolor='white')
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.BORDERS)
ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
ax.set_xticks(np.arange(0, 360, 20), crs=crs_latlon)
ax.set_yticks(np.arange(-80, 20, 20), crs=crs_latlon)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_extent(extent, crs=crs_latlon)
ax.tick_params(labelsize=7)
plt.tight_layout()
if save:
    plt.savefig(out_dir + 'mapa_regiones.pdf', dpi=dpi, bbox_inches='tight')
else:
    plt.show()
################################################################################
print('#######################################################################')
print('done')
print('out_dir = ' + out_dir )
print('#######################################################################')
################################################################################
