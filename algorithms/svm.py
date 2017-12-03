from sklearn.svm import SVC
import numpy as np


def train(x_data,y_data,classes):

    x_data = x_data[:100]
    y_data = y_data[:100]

    svc = SVC(C=10,kernel='poly')

    svc.fit(x_data,y_data)

    return svc

def classify(svc, data):

    data = np.array([data]).reshape(1, -1)

    labels = svc.predict(data)

    return labels[0]