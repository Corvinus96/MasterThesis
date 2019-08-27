from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, BatchNormalization, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D

# Architecture taken from https://github.com/geifmany/cifar-vgg
# Weight decay and Dropout have been removed
def VGG16_Vanilla(input_shape, num_classes):
    model = Sequential([
        
        #0
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        Activation('relu'),
        BatchNormalization(),
        
        #3
        Conv2D(64, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #7
        Conv2D(128, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #10
        Conv2D(128, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #14
        Conv2D(256, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #17
        Conv2D(256, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #20
        Conv2D(256, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #24
        Conv2D(512, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #27
        Conv2D(512, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #30
        Conv2D(512, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #34
        Conv2D(512, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #37
        Conv2D(512, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        #40
        Conv2D(512, (3, 3), padding='same'),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        #45
        Dense(512),
        Activation('relu'),
        BatchNormalization(),
        
        #48
        Dense(num_classes),
        Activation('softmax')
    ])
    return model 

# Architecture taken from https://github.com/geifmany/cifar-vgg
def VGG16(input_shape, num_classes, weight_decay=0.0):
    
    model = Sequential([
        
        #0
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.3),
        
        #4
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #8
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #12
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #16
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #20
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #24
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #28
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #32
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #36
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #40
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #44
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        #Dropout(0.4),
        
        #48
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        
        Flatten(),
        #54
        Dense(512, kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('relu'),
        BatchNormalization(),
        
        #Dropout(0.5),
        #58
        Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay)),
        Activation('softmax')
    ])

    return model

# Architecture taken from https://github.com/geifmany/cifar-vgg
# BatchNormalization before activations
def VGG16_beta(input_shape, num_classes, weight_decay):
    
    model = Sequential([
        
        #0
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.3),
        
        #4
        Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #8
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #12
        Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #16
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #20
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #24
        Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #28
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #32
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #36
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #40
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #44
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        #Dropout(0.4),
        
        #48
        Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        #Dropout(0.5),
        
        Flatten(),
        #54
        Dense(512, kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #Dropout(0.5),
        #58
        Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay)),
        BatchNormalization(scale=False, center=False),
        Activation('softmax')
    ])

    return model

# Architecture taken from https://github.com/geifmany/cifar-vgg
# Weight decay and Dropout have been removed
# BatchNormalization before activations
def VGG16_Vanilla_beta(input_shape, num_classes):
    model = Sequential([
        
        #0
        Conv2D(64, (3, 3), padding='same', input_shape=input_shape),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #3
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #7
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #10
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #14
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #17
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #20
        Conv2D(256, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #24
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #27
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #30
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        #34
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #37
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #40
        Conv2D(512, (3, 3), padding='same'),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        #45
        Dense(512),
        BatchNormalization(scale=False, center=False),
        Activation('relu'),
        
        #48
        Dense(num_classes),
        BatchNormalization(scale=False, center=False),
        Activation('softmax')
    ])
    return model

if __name__ == "__main__":
    model = VGG16((32,32,3), 10, 0.005)