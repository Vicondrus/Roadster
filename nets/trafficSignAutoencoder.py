import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, ReLU, BatchNormalization, Flatten, Dense, \
    Reshape, Conv2DTranspose, UpSampling2D, Activation, LeakyReLU
from tensorflow.keras import backend as K


# architecture inspired by
# https://www.pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/
class TrafficSignNet_Autoencoder_v1:
    @staticmethod
    def build(width, height, depth, filters=((6, 16, 4), (16, 4, 2)), latentDim=16):
        inputShape = (height, width, depth)
        chanDim = -1

        inputs = Input(shape=inputShape)
        x = inputs

        for f, ks, ps in filters:
            x = Conv2D(f, (ks, ks,), padding="same")(x)
            # max pooling not included in autoenc arch 1
            x = MaxPooling2D((ps, ps), padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        encoder = Model(inputs, latent, name="encoder")

        print(encoder.summary())

        latentInputs = Input(shape=(latentDim,))
        x = Dense(np.prod(volumeSize[1:]))(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f, ks, ps in filters[::-1]:
            x = Conv2DTranspose(f, (ks, ks,), padding="same")(x)
            x = UpSampling2D((ps, ps))(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        x = Conv2DTranspose(depth, (1, 1,))(x)
        outputs = Activation("sigmoid")(x)

        decoder = Model(latentInputs, outputs, name="decoder")

        print(decoder.summary())

        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        return encoder, decoder, autoencoder

    @staticmethod
    def clone_encoder(model, width, height, depth, filters=((64, 2), (32, 2)),
                      latentDim=16):
        inputShape = (height, width, depth)
        chanDim = -1

        inputs = Input(shape=inputShape)
        x = inputs

        for (i, (f, ps)) in enumerate(filters):
            x = Conv2D(f, (3, 3,), padding="same", weights=model.layers[1 + i * 3].
                       get_weights())(x)
            # max pooling not included in autoenc arch 1
            # x = MaxPooling2D((ps, ps), padding="same")(x)
            x = ReLU()(x)
            x = BatchNormalization(axis=chanDim)(x)

        volumeSize = K.int_shape(x)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        encoder = Model(inputs, latent, name="encoder")

        return encoder
