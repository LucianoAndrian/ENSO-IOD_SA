"""
Esquema de como funciona el area entre las pdf

"""
# ---------------------------------------------------------------------------- #
save = True
out_dir = '/home/luciano.andrian/doc/ENSO_IOD_SA/salidas/'
name_fig = 'sup_esquema_pdf'
# ---------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Funciones import AreaBetween

if save:
    dpi = 300
else:
    dpi = 100
# ---------------------------------------------------------------------------- #
def normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x - mu) / sigma) ** 2)

x = np.linspace(-5, 5, 100)

# PDFs
y1 = normal_pdf(x, mu=0.0, sigma=1.0)
y2 = normal_pdf(x, mu=0.7, sigma=1.0)

curva1 = pd.Series(data=y1, index=x)
curva2 = pd.Series(data=y2, index=x)

area = AreaBetween(curva1, curva2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y1, label="Climatology", linewidth=2, color='k')
ax.plot(x, y2, label="Event Case", linewidth=2, color='r')
ax.fill_between(x, y1, y2, alpha=0.3, color='orange')
ax.set_title(f"Área between = {area:.4f}")
ax.grid()
ax.set_xlim(-4, 4)
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend(frameon=False)
plt.tight_layout()
if save:
    plt.savefig(f"{out_dir}{name_fig}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()
else:
    plt.show()
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #