#just import the packages that you are gonna use
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.models import save_model

class setModel:
    @staticmethod
    def buildModel(height,width,depth,classes):
        #initilaze the width and height along with the shape to be
        #channels last and the channels dimesnion itself
        #and set the number of classes that you are going to make model from
        model = Sequential()

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            # it doesnt matter so much but if you have channel first Conv layer set to 1
            chanDim = 1
        else:
        # Keras supports "channels_last" and "channel_first"
            inputShape = (height, width, depth)
            chanDim = -1  # this will work when you chose to use channel_last

    #Activation : the method that you are using to train your dataset
    #Batch Normalization: you are making dataset values between [0-1]
    #Max Pooling: Think like a scan method
    #Droput: The procces of disconnecting random neurons between layers


    # CONV => RELU => * 1 POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 2 => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # (CONV => RELU) * 3 => POOL layer set
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))


        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(64))  # Dense means how many neurons
        model.add(Dense(128))
        model.add(Dense(256))
        model.add(Dense(128))
        model.add(Dense(64))
        model.add(Activation("relu"))  # this relu is the like default and changing that u can decide which one to use Activ.
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        #Final Touch
        model.add(Dense(classes))
        model.add(Activation("softmax"))#this is like a prob dist.

        print("THE MODEL HAS SETTED")

        return model


