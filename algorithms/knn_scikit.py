from sklearn.neighbors import KNeighborsClassifier


def train(x_data,y_data,classes):

    print("Begin training")

    # percent10 = int(len(x_data)*0.1)

    # xPercent = x_data[:percent10]
    # yPercent = y_data[:percent10]


    knn = KNeighborsClassifier(n_neighbors=5,p=1,weights='uniform')
    knn.fit(x_data,y_data)

    return knn

def classify(knn,data,labels):

    print("Classify")

    # percent10 = int(len(data)*0.1)
    knn.predict(data)

    return knn.score(data,labels)
