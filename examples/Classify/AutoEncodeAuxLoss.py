import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.utils import to_categorical


def main():
    # this is the size of our encoded representations
    encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    input_img = Input(shape=(784,))
    # "encoded" is the encoded representation of the input
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)

    # "decoded" is the lossy reconstruction of the input
    encoded_img = Input(shape=(32,))
    decoded = Dense(64, activation='relu')(encoded_img)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='sigmoid')(decoded)

    # decode the latent form to the class.
    num_classes = 10
    decode_cls = Dense(64, activation='relu')(encoded_img)
    decode_cls = Dense(128, activation='relu')(decode_cls)
    decode_cls = Dense(256, activation='relu')(decode_cls)
    decode_cls = Dense(num_classes, activation='softmax')(decode_cls)

    # Map the raw input image to "bottleneck" latent representation
    encoder = Model(input_img, encoded)

    # Map the latent representation back to the raw image
    decoder = Model(input=encoded_img, output=decoded)

    # Map the latent form to the class.
    decoder_cls = Model(input=encoded_img, output=decode_cls)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=input_img,
                        output=[decoder(encoder.output), decoder_cls(encoder.output)])

    autoencoder.compile(optimizer='adadelta', loss=['binary_crossentropy', 'categorical_crossentropy'])

    print(autoencoder.summary())
    print(decoder.summary())
    print(decoder_cls.summary())

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    # Convert labels to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    print(x_train.shape)
    print(x_test.shape)

    autoencoder.fit(x=x_train,
                    y=[x_train, y_train],
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, [x_test, y_test]))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(x_test)
    decoded_imgs = decoder.predict(encoded_imgs)

    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display encoding
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(encoded_imgs[i].reshape(4, 8))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(3, n, i + 1 + (2 * n))
        plt.imshow(decoded_imgs[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

    return


if __name__ == "__main__":
    main()
