import numpy as np
import keras.backend as K
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical

def normalize(x, mean, std):
    # This function normalizes inputs for zero mean and unit variance to speed up learning.
    
    # In case std = 0, we add eps = 1e-7
    eps = K.epsilon()
    x = (x-mean)/(std+eps)
    return x

def import_cifar(dataset = 10):
    if dataset == 10:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    elif dataset == 100:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

    # By default, cifar contains 50,000 training and 10,000 test examples
    # We split the 10,000 test data in to 5000 validation and 5000 test
    # The resulting split will be 83% / 8.3% / 8.3% 
    (x_valid, x_test) = np.split(x_test, 2, axis=0)
    (y_valid, y_test) = np.split(y_test, 2, axis=0)
    
    # By default, they are uint8 but we need them float to normalize them
    x_train = x_train.astype('float32')
    x_valid = x_valid.astype('float32')
    x_test  = x_test.astype('float32')
    
    # Calculating the mean and standard deviation of the training data
    mean = np.mean(x_train,axis=(0,1,2,3))
    std = np.std(x_train, axis=(0,1,2,3))
    
    # Normalizing 
    x_train = normalize(x_train, mean, std)
    x_valid = normalize(x_valid, mean, std)
    x_test = normalize(x_test, mean, std)

    y_train = to_categorical(y_train, num_classes=dataset)
    y_valid = to_categorical(y_valid, num_classes=dataset)
    y_test  = to_categorical(y_test, num_classes=dataset)
    
    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

if __name__ == "__main__":
    x_train, y_train, x_valid, y_valid, x_test, y_test = import_cifar()