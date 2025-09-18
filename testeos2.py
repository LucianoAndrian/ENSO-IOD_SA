import numpy as np
import xarray as xr
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# Simulation parameters
ntime, nlat, nlon = 66, 89, 180
rng = np.random.default_rng(0)

# Generate synthetic anomaly field (mean 0, std 1)
data = rng.standard_normal((ntime, nlat, nlon))

# -------- Monte‑Carlo experiment: how many “significant” grid points appear by chance? --------
reps = 200
sig_fractions = []
sig_counts = []

for _ in range(reps):
    events = rng.choice(ntime, size=7, replace=False)
    non_events = np.setdiff1d(np.arange(ntime), events)

    t_stat, p_vals = ttest_ind(
        data[events], data[non_events], axis=0, equal_var=False,
        nan_policy="propagate"
    )

    sig_mask = p_vals < 0.05  # 5 % significance threshold
    sig_fractions.append(sig_mask.mean())
    sig_counts.append(sig_mask.sum())

# Plot histogram of the fraction of grid points flagged as “significant”
plt.figure()
plt.hist(sig_fractions, bins=20, edgecolor="black", facecolor='#FFBE00')
plt.axvline(0.05, linestyle="--", linewidth=2, label="Valor esperado (α = 0.05)")
plt.xlabel("Fracción de píxeles significativos")
plt.ylabel("Frecuencia en 200 composiciones aleatorias")
plt.title("Falsos positivos con solo 7 eventos\n(16 020 píxeles por prueba)")
plt.legend()
plt.tight_layout()
plt.show()

# -------- Visualise the spatial pattern for a single random composite --------
events = rng.choice(ntime, size=7, replace=False)
non_events = np.setdiff1d(np.arange(ntime), events)
_, p_vals = ttest_ind(
    data[events], data[non_events], axis=0, equal_var=False,
    nan_policy="propagate"
)
sig_mask = p_vals < 0.05

plt.figure()
plt.imshow(sig_mask.astype(int), origin="lower", aspect="auto")
plt.xlabel("Longitude index")
plt.ylabel("Latitude index")
plt.title("Example of random 'significant' patches\n(p < 0.05) with n = 7")
plt.colorbar(label="1 = flagged significant")
plt.show()

# Print some basic numbers for reference
print(
    f"Average number of grid points flagged significant (α=0.05): "
    f"{int(np.mean(sig_counts))} ± {int(np.std(sig_counts))}"
)

