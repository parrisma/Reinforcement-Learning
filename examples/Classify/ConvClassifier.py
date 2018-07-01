from keras import backend as K
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical


def main():
    num_classes = 10
    irws = 28
    icls = 28
    batch_size = 256

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and note input shape
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, irws, icls)
        x_test = x_test.reshape(x_test.shape[0], 1, irws, icls)
        input_shape = (1, irws, icls)
    else:
        x_train = x_train.reshape(x_train.shape[0], irws, icls, 1)
        x_test = x_test.reshape(x_test.shape[0], irws, icls, 1)
        input_shape = (irws, icls, 1)

    # Re Scale 0 - 1
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    # Convert labels to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Construct 2D Convolution Model
    # 5 x 5, stride 1 and 32 Channels
    # Max Pool 2x2, stride 2
    # 5 x 5, stride 1 and 64 Channels
    # Max Pool 2x2 stride 1
    # Flatten to vector as input to Dense layer
    # Softmax
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile Model
    model.compile(loss=categorical_crossentropy,
                  optimizer=SGD(lr=0.01),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              epochs=50,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, y_test))
    return


if __name__ == "__main__":
    main()
