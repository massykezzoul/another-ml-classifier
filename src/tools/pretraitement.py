import numpy as np

def normalize_dataset(x_train, y_train):
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / np.amax(x_train)

    # mean and standart deviation
    mean = np.mean(x_train)
    std = np.std(x_train)
    x_train = (x_train - mean) / std

    # Flatten the images.
    #x_train = x_train.reshape((-1, len(x_train[0]) * len(x_train[0][0])))
    # One hot encoding
    unique_y = np.sort(np.unique(y_train))
    y_train = np.array(list(map(lambda x:  [1 if x == k else 0 for k in unique_y], y_train)))

    return x_train , y_train
