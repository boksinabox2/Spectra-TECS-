# PCA to discriminate between the different concentrations of lactose in milk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA as pca
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.cluster import KMeans
from hyperparameters import hyperparameters

# Import data from csv on github
url = hyperparameters["data"]["url"]
data = pd.read_csv(url)
 
# The first column of the Data Frame contains the labels
lab = data.values[:, hyperparameters["data"]["label_column"]].astype('uint8') 
 
# Read the features (scans) and transform data from reflectance to absorbance
feat = np.log(1.0 / (data.values[:, hyperparameters["data"]["feature_columns"]]).astype('float32'))
 
# Calculate first derivative applying a Savitzky-Golay filter
dfeat = savgol_filter(
    feat,
    hyperparameters["savgol"]["window_length"],
    polyorder=hyperparameters["savgol"]["polyorder"],
    deriv=hyperparameters["savgol"]["derivative"]
)

# Feature selection - how many do I need?

# Initialise
nc = hyperparameters["pca"]["n_components"]
pca1 = pca(n_components=nc)
pca2 = pca(n_components=nc)
 
# Scale the features to have zero mean and standard deviation of 1
# This is important when correlating data with very different variances
nfeat1 = StandardScaler().fit_transform(feat)
nfeat2 = StandardScaler().fit_transform(dfeat)
 
# Fit the spectral data and extract the explained variance ratio
X1 = pca1.fit(nfeat1)
expl_var_1 = X1.explained_variance_ratio_
 
# Fit the first data and extract the explained variance ratio
X2 = pca2.fit(nfeat2)
expl_var_2 = X2.explained_variance_ratio_
 
# Running the Classification of NIR spectra using Principal Component Analysis
pca2 = pca(n_components=hyperparameters["classification"]["n_pcs_to_use"])  # Change depending on how many PCs to use
 
# Transform on the scaled features
Xt2 = pca2.fit_transform(nfeat2)

# Labels for the plot legend
labplot = hyperparameters["plotting"]["labels"]
 
# Scatter plot
unique = list(set(lab))
colors = [plt.cm.jet(float(i) / max(unique)) for i in unique]
with plt.style.context(('ggplot')):
    plt.figure(figsize=(8, 6))
    for i, u in enumerate(unique):
        col = np.expand_dims(np.array(colors[i]), axis=0)
        xi = [Xt2[j, 0] for j in range(len(Xt2[:, 0])) if lab[j] == u]
        yi = [Xt2[j, 1] for j in range(len(Xt2[:, 1])) if lab[j] == u]
        plt.scatter(xi, yi, c=col, s=hyperparameters["plotting"]["point_size"], edgecolors='k', label=str(u))
 
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(labplot, loc='upper right')
    plt.title('Principal Component Analysis')
    plt.show()