from keras import Input, Model
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Concatenate, \
    Dense, Lambda, Reshape, UpSampling2D, Conv2DTranspose


class TrafficSignNet_Variational_Autoencoder:
    mean = None
    var = None

    @staticmethod
    def _sampling(args):
        mean, log_var = args
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.0)
        return mean + K.exp(log_var) * epsilon

    @staticmethod
    def _cache_mean(arg):
        TrafficSignNet_Variational_Autoencoder.mean = arg
        return arg

    @staticmethod
    def _cache_var(arg):
        TrafficSignNet_Variational_Autoencoder.var = arg
        return arg

    @staticmethod
    def build_and_compile(width, height, depth, hidden=2):
        input_img = Input(shape=(width, height, depth))
        encoder = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)

        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(
            encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)
        encoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(
            encoder_branch_left)
        encoder_branch_left = MaxPooling2D((2, 2), padding='same')(encoder_branch_left)

        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(encoder)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same')(
            encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(
            encoder_branch_right)
        encoder_branch_right = Conv2D(16, (3, 3), activation='relu', padding='same')(
            encoder_branch_right)
        encoder_branch_right = AveragePooling2D((2, 2), padding='same')(
            encoder_branch_right)

        encoder_out = Flatten()(
            Concatenate()([encoder_branch_left, encoder_branch_right]))
        encoder_out = Dense(128, activation='relu')(encoder_out)

        encoder = Model(input_img, encoder_out, name="encoder")

        print(encoder.summary())

        r_input = Input(shape=(128,), name="r_input")
        mean = Dense(hidden, name='mean')(r_input)
        log_var = Dense(hidden, name='log_var')(r_input)

        mirror = Lambda(TrafficSignNet_Variational_Autoencoder._sampling)([mean,
                                                                           log_var])

        R = Model(r_input, mirror, name="R")
        print(R.summary())

        inputDec = Input(shape=(2,))

        decoder = Dense(128, activation='relu')(inputDec)
        decoder = Dense(16 * 4 * 4, activation='relu')(decoder)
        decoder = Reshape((4, 4, 16))(decoder)

        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(
            decoder)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(
            decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu')(decoder_branch_left)
        decoder_branch_left = UpSampling2D((2, 2))(decoder_branch_left)
        decoder_branch_left = Conv2D(16, (3, 3), activation='relu', padding='same')(
            decoder_branch_left)

        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu')(decoder)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu')(
            decoder_branch_right)
        decoder_branch_right = UpSampling2D((2, 2))(decoder_branch_right)
        decoder_branch_right = Conv2DTranspose(16, (3, 3), activation='relu',
                                               padding='same')(decoder_branch_right)

        out = Concatenate()([decoder_branch_left, decoder_branch_right])
        out = Conv2D(16, (3, 3), activation='relu', padding='same')(out)
        out_img = Conv2D(depth, (3, 3), activation='sigmoid', padding='same')(out)

        decoder = Model(inputDec, out_img, name="decoder")

        print(decoder.summary())

        autoencoder = Model(input_img, decoder(R(encoder(input_img))),
                            name="autoencoder")

        print(autoencoder.summary())

        autoencoder.compile(optimizer='rmsprop', loss='mse')

        return autoencoder
