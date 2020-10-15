from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, ReLU, BatchNormalization, \
    Conv2DTranspose, UpSampling2D, Activation
from tensorflow.keras import backend as K


# architecture inspired by
# https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/
class TrafficSignNet_Autoencoder_v2:
    @staticmethod
    def build(width, height, depth, filters=((16, 4, 4), (8, 4, 2), (3, 2, 2))):
        inputShape = (height, width, depth)
        chanDim = -1

        inputs = Input(shape=inputShape)
        x = inputs

        for f, ks, ps in filters:
            x = Conv2D(f, (ks, ks,), padding="same")(x)
            # max pooling not included in autoenc arch 1
            x = MaxPooling2D((ps, ps), padding="same")(x)
            x = ReLU()(x)
            x = BatchNormalization(axis=chanDim)(x)

        encoded = x

        encoder = Model(inputs, encoded, name="encoder")

        print(encoder.summary())

        volumeSize = K.int_shape(x)
        decInput = Input(shape=(volumeSize[1], volumeSize[2], volumeSize[3],))
        x = decInput

        for f, ks, ps in filters[::-1]:
            x = Conv2DTranspose(f, (ks, ks,), padding="same")(x)
            x = UpSampling2D((ps, ps))(x)
            x = ReLU()(x)
            # x = BatchNormalization(axis=chanDim)(x)

        x = Conv2DTranspose(depth, (1, 1,))(x)
        outputs = Activation("sigmoid")(x)

        decoder = Model(decInput, outputs, name="decoder")

        print(decoder.summary())

        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        return encoder, decoder, autoencoder
