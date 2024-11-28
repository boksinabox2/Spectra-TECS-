#wavelet transform for baseline correction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt # Needed for wavelet decomposition

#reading data from the excel
data = pd.read_excel("C:/NIRsampledatarocks/NIR data Earth analogs/NIR data Earth analogs.xlsx")
absorbance = data['C1'].values
wavelength = data['nm'].values
interval = np.arange(500,600,1)

wavelet_type = 'sym4' #change wavelet type based on how noisy/type of data
# Wavelet transform
coeffs_NIRS = pywt.wavedec(absorbance, wavelet_type, level = 7) #change the level here too depending on requirement
 
# Inverse wavelet transform
alpha = 0.9 #change depending on how much of the baseline I want to retain versus suppress. (A lower alpha (closer to 0) will suppress more of the baseline, while a higher alpha (closer to 1) will retain more of it.)
new_coeffs_NIRS = coeffs_NIRS.copy()
new_coeffs_NIRS[0] *= alpha
 
# Inverse wavelet transform
new_NIRS = pywt.waverec(new_coeffs_NIRS, wavelet_type)
plt.figure(figsize=(9,6))
plt.plot(wavelength[interval], absorbance[interval], label = 'Original Spectrum')
plt.plot(wavelength[interval], new_NIRS[interval], label = 'Baseline correction with wavelet decomposition')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity")
plt.title("NIR spectrum")
plt.legend()
plt.show()