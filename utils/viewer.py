"""File for viewing inputs"""

def view_img(data_line):
    """view data_line as an image"""

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.imshow(data_line.reshape((28, 28)), interpolation='nearest', cmap='gray')
    plt.grid(True)
    plt.show()
