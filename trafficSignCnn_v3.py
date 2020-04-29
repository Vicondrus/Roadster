from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential

hidden_num_units = 2048
hidden_num_units1 = 1024
hidden_num_units2 = 128
pool_size = (2, 2)


# architecture by Sanket Doshi
# can be found at https://towardsdatascience.com/traffic-sign-detection-using-convolutional-neural-network-660fb32fe90e
class TrafficSignNet_v3:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (width, height, depth)
        model = Sequential([Conv2D(16, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
                            BatchNormalization(), Conv2D(16, (3, 3), activation='relu', padding='same'),
                            BatchNormalization(),
                            MaxPooling2D(pool_size=pool_size),
                            Dropout(0.2),

                            Conv2D(32, (3, 3), activation='relu', padding='same'),
                            BatchNormalization(),

                            Conv2D(32, (3, 3), activation='relu', padding='same'),
                            BatchNormalization(),
                            MaxPooling2D(pool_size=pool_size),
                            Dropout(0.2),

                            Conv2D(64, (3, 3), activation='relu', padding='same'),
                            BatchNormalization(),

                            Conv2D(64, (3, 3), activation='relu', padding='same'),
                            BatchNormalization(),
                            MaxPooling2D(pool_size=pool_size),
                            Dropout(0.2), Flatten(), Dense(units=hidden_num_units, activation='relu'),
                            Dropout(0.3),
                            Dense(units=hidden_num_units1, activation='relu'),
                            Dropout(0.3),
                            Dense(units=hidden_num_units2, activation='relu'),
                            Dropout(0.3),
                            Dense(units=classes, input_dim=hidden_num_units, activation='softmax'),
                            ])
        return model
