#1 using PCA and mahalanobis distance to detect outliers

#load and plot data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# Absorbance data, collected in the matrix X
data = pd.read_csv('https://raw.githubusercontent.com/nevernervous78/nirpyresearch/master/data/plums.csv').values[:,1:]
X = np.log(1.0/data)
wl = np.arange(1100,2300,2)
 
# Plot the data
# fig = plt.figure(figsize=(8,6))
# with plt.style.context(('ggplot')):
#     plt.plot(wl, X.T)
#     plt.xlabel('Wavelength (nm)')
#     plt.ylabel('Absorbance spectra')
#     plt.show()

#making score plot with first two principal component
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
 
# Define the PCA object
pca = PCA()
 
# Run PCA on scaled data and obtain the scores array
T = pca.fit_transform(StandardScaler().fit_transform(X))
 
# Score plot of the first 2 PC
# fig = plt.figure(figsize=(8,6))
# with plt.style.context(('ggplot')):
#     plt.scatter(T[:, 0], T[:, 1], edgecolors='k', cmap='jet')
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.title('Score Plot')
# plt.show()

#Euclidean distance? perchance? (experiment)

#redraw score plot
# fig = plt.figure(figsize=(8,6))
# with plt.style.context(('ggplot')):
#     plt.scatter(T[:, 0], T[:, 1], edgecolors='k', cmap='jet')
#     plt.xlim((-60, 60))
#     plt.ylim((-60, 60))
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.title('Score Plot')
# plt.show()

# Compute the euclidean distance using the first 5 PC
euclidean = np.zeros(X.shape[0])
for i in range(5): #change dep on how many PC to use
    euclidean += (T[:,i] - np.mean(T[:,:5]))**2/np.var(T[:,:5])
    
#calculate euclidean distance all at once + fancy colour coded score plot
# colors = [plt.cm.jet(float(i)/max(euclidean)) for i in euclidean]
# fig = plt.figure(figsize=(8,6))
# with plt.style.context(('ggplot')):
#     plt.scatter(T[:, 0], T[:, 1], c=colors, edgecolors='k', s=60)
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.xlim((-60, 60))
#     plt.ylim((-60, 60))
#     plt.title('Score Plot')
# plt.show()

#using mahalanobis distance (more effective/less margin for error)

from sklearn.covariance import EmpiricalCovariance, MinCovDet
 
# fit a Minimum Covariance Determinant (MCD) robust estimator to data 
robust_cov = MinCovDet().fit(T[:,:5])
 
# Get the Mahalanobis distance
m = robust_cov.mahalanobis(T[:,:5])

#colour score plot using mahalanobis instead
colors = [plt.cm.jet(float(i)/max(m)) for i in m]
fig = plt.figure(figsize=(8,6))
with plt.style.context(('ggplot')):
    plt.scatter(T[:, 0], T[:, 1], c=colors, edgecolors='k', s=60)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim((-60, 60))
    plt.ylim((-60, 60))
    plt.title('Score Plot')
plt.show()