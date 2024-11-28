hyperparameters = {
# for PCA1
    "data": {
        "url": "https://raw.githubusercontent.com/nevernervous78/nirpyresearch/master/data/milk.csv",
        "label_column": 1,  # Column index for labels
        "feature_columns": slice(2, None)  # Columns for features (scans)
    },
    "savgol": {
        "window_length": 25,  # Length of the filter window
        "polyorder": 5,       # Polynomial order for the filter
        "derivative": 1       # Derivative order
    },
    "pca": {
        "n_components": 10  # Number of components for PCA
    },
    "classification": {
        "n_pcs_to_use": 4  # Number of PCs to use in classification
    },
    "plotting": {
        "labels": [
            "0/8 Milk", "1/8 Milk", "2/8 Milk", "3/8 Milk",
            "4/8 Milk", "5/8 Milk", "6/8 Milk", "7/8 Milk", "8/8 Milk"
        ],
        "point_size": 60  # Size of scatter plot points
    },
#for savitzky golay
    'smoothing_window': 5,  # Width of the selection window for Savitzky-Golay filter
    'polynomial_order': 2,   # Order of the polynomial for Savitzky-Golay filter
    'file_path': "C:/NIRsampledatarocks/NIR data Earth analogs/NIR data Earth analogs.xlsx",  # URL of the dataset
    "interval": {
        "start": 500,
        "end": 600,
        "step": 1
    },
#for wavelettransform
    'file_path_wavelet': "C:/NIRsampledatarocks/NIR data Earth analogs/NIR data Earth analogs.xlsx",  # URL of the dataset
    "wavelet_transform": {
        "wavelet_type": 'sym4',
        "level": 7,
        "alpha": 0.9
    },
#for outlier detection
    'data': 'https://raw.githubusercontent.com/nevernervous78/nirpyresearch/master/data/plums.csv', 
    'n_components': 5
    
    }


                                
 
