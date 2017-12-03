""" View image by image from the dataset """

# Univariate Histograms
import matplotlib.pyplot as plt
import pandas
import utils.reader
import numpy as np
from pandas.plotting import scatter_matrix

def main():
    """ The main function """

    def transform(a):
        """Average first and last element of a 1-D array"""
        return np.array([np.mean(a), np.std(a)])

    images, labels = utils.reader.load_train('./data')
    images = np.apply_along_axis(transform, 1, images)
    print(images.shape)

    dataset = np.zeros((images.shape[0], images.shape[1] + 1))
    dataset[:,:-1] = images
    dataset[:,-1] = labels
    data = pandas.DataFrame(data=dataset)
    data.groupby([dataset.shape[1]-1])
    data["target"] = labels
    # data.hist(column=[300,301])
    print('Plotting...')
    color_wheel = { 1: "#e6194b", 
                    2: "#3cb44b", 
                    3: "#ffe119", 
                    4: "#0082c8", 
                    5: "#f58231", 
                    6: "#911eb4", 
                    7: "#46f0f0", 
                    8: "#f032e6", 
                    9: "#d2f53c", 
                    0: "#fabebe"}
    colors = data["target"].map(lambda x: color_wheel.get(x))
    scatter_matrix(data[[0, 1]], color=colors, alpha=0.6, figsize=(5, 5), diagonal='hist')
    plt.show()

if __name__ == "__main__":
    main()
