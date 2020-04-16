import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense


# learning transfer from MobileNetV2
#   learning transfer using different base network
class TrafficSignNet_v2:
    @staticmethod
    def build(width, height, depth, classes):
        IMG_SHAPE = (width, height, 3)
        chanDim = -1
        base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

        base_model.trainable = False

        # fine_tune_at = 80 * len(base_model.layers) / 100

        #for layer in base_model.layers[:int(fine_tune_at)]:
        #    layer.trainable = False

        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        model = tf.keras.Sequential()

        model.add(base_model)

        model.add(global_average_layer)

        model.add(Dropout(0.3))
        model.add(Dense(classes, kernel_regularizer=tf.keras.regularizers.l2(0.01)))
        model.add(Activation("softmax"))

        return model
