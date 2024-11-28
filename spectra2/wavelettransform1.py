import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt  # Needed for wavelet decomposition

# Import hyperparameters
from hyperparameters import hyperparameters

url = hyperparameters['file_path_wavelet']  # URL of the data
data = pd.read_excel(url)  # Use the URL from the hyperparameters

# Extract the relevant columns
absorbance = data['C1'].values
wavelength = data['nm'].values

# Get the interval from the hyperparameters
start = hyperparameters["interval"]["start"]
end = hyperparameters["interval"]["end"]
step = hyperparameters["interval"]["step"]

# Create the interval using the hyperparameters
interval = np.arange(start, end, step)

# Get the wavelet transform parameters from the hyperparameters
wavelet_type = hyperparameters["wavelet_transform"]["wavelet_type"]  # Wavelet type
level = hyperparameters["wavelet_transform"]["level"]              # Decomposition level
alpha = hyperparameters["wavelet_transform"]["alpha"]              # Baseline correction strength

# Wavelet transform
coeffs_NIRS = pywt.wavedec(absorbance, wavelet_type, level=level)

# Inverse wavelet transform
new_coeffs_NIRS = coeffs_NIRS.copy()
new_coeffs_NIRS[0] *= alpha  # Adjust the baseline according to alpha

# Inverse wavelet transform
new_NIRS = pywt.waverec(new_coeffs_NIRS, wavelet_type)

# Plot the results
plt.figure(figsize=(9,6))
plt.plot(wavelength[interval], absorbance[interval], label='Original Spectrum')
plt.plot(wavelength[interval], new_NIRS[interval], label='Baseline correction with wavelet decomposition')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("NIR spectrum")
plt.legend()
plt.show()