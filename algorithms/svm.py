from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import utils.viewer
from scipy.ndimage import rotate
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift
from skimage.transform import rescale
# (60k) C=10, kernel='poly' -> 87.23%

# ----------------------------------- WRONG

# C=10, kernel='poly'
# SIZE | NORMAL | W/ [-10,10] ROTATIONS
# 100 | 43.69 | 53.55
# 1000 | 59.10 | 66.56
# 60k | 87.23 | 

# DOWNSCALE to 7x7
# C=10, kernel='poly'
# SIZE | NORMAL | W/ [-10,10] ROTATIONS
# 100 | 43.69 | 52.52
# 1000 | 59.10 | 67.38
# 60k | 87.23 | 

# ----------------------------------- RIGHT

# normalized -> downscale -> rotate
# DOWNSCALE to 7x7, -10,10 rotations, normalized by feature
# C=10, kernel='poly'
# SIZE | NORMAL | THIS
# 100 | 43.69 | 65.74
# 1000 | 59.10 | 75.79%
# 5000 | ? | 80.27
# 60k | 87.23 | 84.91


# ----------------------------------- RIGHT

# normalized -> downscale -> rotate
# DOWNSCALE to 14x14, -5,5 rotations, normalized by feature
# C=10, kernel='poly'
# SIZE | NORMAL | THIS
# 100 | 43.69 | 64.34
# 1000 | 59.10 | 79.14
# 5000 | ? | 84.63
# 60k | 87.23 | 89.39


# ----------------------------------- RIGHT

# downscale ->rotate -> normalize
# DOWNSCALE to 14x14, -5,5 rotations, normalized by feature
# C=10, kernel='poly'
# SIZE | NORMAL | THIS
# 100 | 43.69 | 64.21
# 1000 | 59.10 | 78.56
# 5000 | ? | 83.91%
# 60k | 87.23 | 89.53%

# -------------------------------

# nrot = (n + 1) * 20
# n = 1k => nrot = 21k
# n = 60k => nrot = 1260000k

# (5k) C=10**6, kernel='rbf', gamma=(4*10**-7), -> 35.93%
# (60k) normalized C=10**6, kernel='rbf', gamma=(4*10**-7), -> 31.28%

def normalize(a):
    return (a - np.mean(a)) / max(0.001, np.std(a))

def train(x_data,y_data,classes):


    x_data = x_data / 255.0
    y_data = y_data

    print(x_data.shape)

    new_x_data = []
    new_y_data = []

    for x,y in zip(x_data, y_data):
        img = x.reshape((28,28))
        image_rescaled = rescale(img, 1.0 / 2.0)
        # utils.viewer.view_img(img.flatten())
        new_x_data.append(image_rescaled.flatten())
        new_y_data.append(y)
        for da in range(-5, 5):
            rot = rotate(image_rescaled, da, reshape=False)
            new_x_data.append(rot.flatten())
            new_y_data.append(y)
            # utils.viewer.view_img(rot.flatten())

    # utils.viewer.view_img(rot.flatten())
    x_data = np.array(new_x_data)
    y_data = np.array(new_y_data)

    features_means = []
    features_std = []

    for f in range(14*14):
        features_means.append(np.mean(x_data[:,f]))
        features_std.append(max(0.001, np.std(x_data[:,f])))

    x_data = np.apply_along_axis(normalize, 0, x_data)

    print(x_data.shape)

    svc = SVC(C=10**6, kernel='rbf', gamma=4*(10**(-7)), verbose=True)

    svc.fit(x_data,y_data)

    return svc, features_means, features_std

def classify(param, data):

    svc, features_means, features_std = param

    data = data / 255.0

    # data = normalize(data)

    img = data.reshape((28,28))
    image_rescaled = rescale(img, 1.0 / 2.0)
    data = image_rescaled.flatten()

    for f in range(14*14):
        data[f] = (data[f] - features_means[f]) / features_std[f]

    data = np.array([data]).reshape(1, -1)

    labels = svc.predict(data)

    return labels[0]