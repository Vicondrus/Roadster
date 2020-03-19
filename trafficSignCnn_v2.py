import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# learning transfer from MobileNetV2
class TrafficSignNet_v2:
    @staticmethod
    def build(width, height, depth, classes):
        IMG_SHAPE = (width, height, 3)
        chanDim = -1
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                       include_top=False,
                                                       weights='imagenet')

        base_model.trainable = True

        fine_tune_at = 100

        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential([
            base_model,
            global_average_layer
        ])

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model