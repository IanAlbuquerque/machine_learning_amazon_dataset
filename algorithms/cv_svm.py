# 80.36999999%

from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import utils.viewer
from scipy.ndimage import rotate
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift, binary_fill_holes
from skimage.transform import rescale
# import cv2
from skimage.feature import corner_harris, peak_local_max
from skimage import filters
from skimage import exposure
from skimage.segmentation import active_contour
from skimage.segmentation import mark_boundaries, find_boundaries
from skimage.feature import canny
from skimage.morphology import watershed
from skimage.morphology import square, dilation, closing

NUM_FEATURES = 10
BLOCK_SIZE = 2
K_SIZE = 5
K_HYPER = 0.02

def normalize(a):
    return (a - np.mean(a)) / max(0.001, np.std(a))

def train(x_data,y_data,classes):
    global NUM_FEATURES
    global BLOCK_SIZE
    global K_SIZE
    global K_HYPER

    x_data = x_data / 255.0
    y_data = y_data

    print(x_data.shape)

    new_x_data = []
    new_y_data = []

    for x,y in zip(x_data, y_data):
        img = x.reshape((28,28))

        img = canny(img)
        img = closing(img, square(3))
        img = filters.gaussian(img)
        # img = binary_fill_holes(img)
        # img = exposure.equalize_hist(img)
        # img = filters.sobel(img)
        # img = img ** 2
        # img[img > np.max(img)*0.1] = 1.0
        # img[img <= np.max(img)*0.1] = 0.0

        # s = np.linspace(0, 2*np.pi, 400)
        # x = 14 + 7*np.cos(s)
        # y = 14 + 7*np.sin(s)
        # init = np.array([x, y]).T

        # snake = active_contour(filters.gaussian(img, 2),
        #                     init, alpha=0.015, beta=1, gamma=0.001)

        # print(snake)
        # utils.viewer.view_img(img.flatten())

        # sn = np.zeros((28,28))
        # for pt in snake:
        #     sn[int(pt[0]), int(pt[1])] = 1
        # utils.viewer.view_img(sn.flatten())

        #-----V
        # utils.viewer.view_img(img.flatten())
        # img = binary_fill_holes(img)
        #------

        # markers = np.zeros_like(img)
        # markers[img > np.max(img)*0.8] = 1
        # markers[img <= np.max(img)*0.1] = 2
        # utils.viewer.view_img(markers.flatten())

        # img = watershed(img, markers)
        # boundaries = find_boundaries(img)
        # img[boundaries] = 1
        # print(boundaries)
        # img = mark_boundaries(img, boundaries)

        #-----V
        # utils.viewer.view_img(img.flatten())


        # h = corner_harris(img, method='eps', sigma=1.1)
        # utils.viewer.view_img(h.flatten())
        # coords = peak_local_max(h, min_distance=2, exclude_border=False, indices=True, threshold_abs=0, num_peaks = NUM_FEATURES)

        # print(len(coords))
        # disp = np.zeros((28,28))
        # for pt in coords:
        #     disp[pt[0], pt[1]] = 1.0
        # utils.viewer.view_img(disp.flatten())
        #------

        # gray = np.float32(img)
        # dst = cv2.cornerHarris(gray, BLOCK_SIZE, K_SIZE, K_HYPER)

        # idxs =  np.argpartition(dst.flatten(), -NUM_FEATURES)[-NUM_FEATURES:]
        # maxes = np.vstack(np.unravel_index(idxs, dst.shape)).T

        # dst = dst * 0.5
        # dst[dst > np.sort(dst.flatten())[-NUM_FEATURES]] = 1
        # print(maxes.flatten())

        # np.sort(dst.flatten())[-10]
        # print(np.sort(dst.flatten())[-10])
        # print(dst)
        # dst[dst>0.01*dst.max()]=1

        # cv2.imshow('dst',rescale(dst, 10.0))
        # if cv2.waitKey(0) & 0xff == 27:
        #     cv2.destroyAllWindows()
        
        new_x_data.append(img.flatten())
        new_y_data.append(y)

        # image_rescaled = rescale(img, 1.0 / 2.0)
        # utils.viewer.view_img(img.flatten())
        # new_x_data.append(image_rescaled.flatten())
        # new_y_data.append(y)
        for da in range(-5, 5):
            rot = rotate(img, da, reshape=False)
            new_x_data.append(rot.flatten())
            new_y_data.append(y)
            # utils.viewer.view_img(rot.flatten())

    # utils.viewer.view_img(rot.flatten())
    x_data = np.array(new_x_data)
    y_data = np.array(new_y_data)

    features_means = []
    features_std = []

    # for f in range(NUM_FEATURES*2):
    #     features_means.append(np.mean(x_data[:,f]))
    #     features_std.append(max(0.001, np.std(x_data[:,f])))

    # x_data = np.apply_along_axis(normalize, 0, x_data)

    print(x_data.shape)

    svc = SVC(C=10**6, kernel='rbf', gamma=4*(10**(-7)), verbose=True)

    svc.fit(x_data,y_data)

    return svc, features_means, features_std

def classify(param, data):
    global NUM_FEATURES
    global BLOCK_SIZE
    global K_SIZE
    global K_HYPER

    svc, features_means, features_std = param

    data = data / 255.0

    img = data.reshape((28,28))

    img = canny(img)
    # img = dilation(img, square(3))
    img = closing(img, square(3))
    img = filters.gaussian(img)
    data = img.flatten()

    # gray = np.float32(img)
    # dst = cv2.cornerHarris(gray, BLOCK_SIZE, K_SIZE, K_HYPER)

    # idxs =  np.argpartition(dst.flatten(), -NUM_FEATURES)[-NUM_FEATURES:]
    # maxes = np.vstack(np.unravel_index(idxs, dst.shape)).T

    # data = maxes.flatten()

    # img = data.reshape((28,28))
    # image_rescaled = rescale(img, 1.0 / 2.0)
    # data = image_rescaled.flatten()

    # for f in range(NUM_FEATURES*2):
    #     data[f] = (data[f] - features_means[f]) / features_std[f]

    data = np.array([data]).reshape(1, -1)

    labels = svc.predict(data)

    return labels[0]