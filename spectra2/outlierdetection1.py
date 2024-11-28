import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from hyperparameters import hyperparameters  # Import the hyperparameters

# Load the data from the URL specified in the hyperparameters file
url = hyperparameters["data"]  # Use URL from hyperparameters
data = pd.read_csv(url).values[:, 1:]

# Apply log transformation on the data (to get absorbance)
X = np.log(1.0 / data)
wl = np.arange(1100, 2300, 2)

# Define the number of principal components to use for PCA and Mahalanobis distance
n_components = hyperparameters["n_components"]  # Get the number of components from hyperparameters

# Run PCA on scaled data and obtain the scores array
pca = PCA(n_components=n_components)
T = pca.fit_transform(StandardScaler().fit_transform(X))

# Using Mahalanobis distance for outlier detection
robust_cov = MinCovDet().fit(T[:, :n_components])

# Get the Mahalanobis distance
m = robust_cov.mahalanobis(T[:, :n_components])

# Color the score plot based on Mahalanobis distance
colors = [plt.cm.jet(float(i) / max(m)) for i in m]
fig = plt.figure(figsize=(8, 6))

# Plot the score plot
with plt.style.context(('ggplot')):
    plt.scatter(T[:, 0], T[:, 1], c=colors, edgecolors='k', s=60)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.xlim((-60, 60))
    plt.ylim((-60, 60))
    plt.title('Score Plot')
plt.show()