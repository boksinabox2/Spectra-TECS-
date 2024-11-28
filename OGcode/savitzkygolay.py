#savitzky golay smoothing to remove noise
import numpy as np
import pandas as pd # type: ignore
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter  #removed general_gaussian from here # type: ignore

#setting up and reading the data from excel file - practice dataset
data = pd.read_excel("C:/NIRsampledatarocks/NIR data Earth analogs/NIR data Earth analogs.xlsx")
X = data['C1'].values
wl = data['nm'].values

#applying parameters to smooth the data 
w = 5 #width of selection window
p = 2 #order of the polynomial to be fitted to the data
X_smooth_1 = savgol_filter(X, w, polyorder = p, deriv=0) #simple smoothing (no derivatives) if want to use derivatives - set deriv to >0
#to increase amount of smoothing, increase ratio w/p
X_smooth_2 = savgol_filter(X, 2*w+1, polyorder = p, deriv=0)
X_smooth_3 = savgol_filter(X, 4*w+1, polyorder = 3*p, deriv=0)

#plotting an interval of the whole spectra, after applying the different smoothings
plt.figure(figsize=(9,6))
interval = np.arange(500,600,1)
plt.plot(wl[interval], X[interval], 'b', label = 'No smoothing')
plt.plot(wl[interval], X_smooth_1[interval], 'r', label = 'Smoothing: w/p = 2.5')
plt.plot(wl[interval], X_smooth_2[interval], 'g', label = 'Smoothing: w/p = 5.5')
plt.plot(wl[interval], X_smooth_3[interval], 'm', label = 'Smoothing: w/p = 3.5')
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.legend()
plt.show()