from sklearn.svm import SVC


def train(x_data,y_data,classes):

    svc = SVC(C=10,kernel='poly')

    svc.fit(x_data,y_data)

    return svc

def classify(svc,data,labels):

    svc.predict(data)

    return svc.score(data,labels)