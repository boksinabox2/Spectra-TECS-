import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # removed general_gaussian from here # type: ignore

# Import hyperparameters
from hyperparameters import hyperparameters

# Setting up and reading the data from an Excel file - practice dataset
url = hyperparameters['file_path']  # URL of the data
data = pd.read_excel(url)  # Use the URL from the hyperparameters

X = data['C1'].values
wl = data['nm'].values

# Extract hyperparameters
w = hyperparameters['smoothing_window']  # Width of selection window
p = hyperparameters['polynomial_order']  # Order of the polynomial to be fitted to the data

# Applying parameters to smooth the data 
X_smooth_1 = savgol_filter(X, w, polyorder=p, deriv=0)  # Simple smoothing (no derivatives)
X_smooth_2 = savgol_filter(X, 2 * w + 1, polyorder=p, deriv=0)
X_smooth_3 = savgol_filter(X, 4 * w + 1, polyorder=3 * p, deriv=0)

# Plotting an interval of the whole spectra, after applying the different smoothings
plt.figure(figsize=(9, 6))
interval = np.arange(500, 600, 1)
plt.plot(wl[interval], X[interval], 'b', label='No smoothing')
plt.plot(wl[interval], X_smooth_1[interval], 'r', label=f'Smoothing: w/p = {w/p}')
plt.plot(wl[interval], X_smooth_2[interval], 'g', label=f'Smoothing: w/p = {2 * w + 1}/{p}')
plt.plot(wl[interval], X_smooth_3[interval], 'm', label=f'Smoothing: w/p = {4 * w + 1}/{3 * p}')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.show()