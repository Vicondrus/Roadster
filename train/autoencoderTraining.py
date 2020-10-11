import argparse
import os

import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.datasets import mnist

from trafficSignAutoencoder import TrafficSignNet_Autoencoder
from util import load_data_and_labels

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--dataset", required=True, help="path to input training model")
# args = vars(ap.parse_args())
#
# trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
# testPath = os.path.sep.join([args["dataset"], "Test.csv"])
#
# trainX, trainY = load_data_and_labels(args["dataset"], trainPath)
# # test data
# testX, testY = load_data_and_labels(args["dataset"], testPath)
#
# # trainX = np.expand_dims(trainX, axis=-1)
# # testX = np.expand_dims(testX, axis=-1)
# trainX = trainX.astype("float32") / 255.0
# testX = testX.astype("float32") / 255.0

enc, dec, autoenc = TrafficSignNet_Autoencoder.build(width=32, height=32, depth=3)
# opt = Adam(lr=1e-3)
autoenc.compile(optimizer='adadelta', loss='mean_squared_error')

EPOCHS = 15
BS = 32

train_datagen = ImageDataGenerator(rescale=1. / 255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory(
    'data/germanRoadsigns2/Train',
    target_size=(32, 32),
    batch_size=BS,
    class_mode='input'
)

test_datagen = ImageDataGenerator(rescale=1. / 255, data_format='channels_last')
validation_generator = test_datagen.flow_from_directory(
    'data/germanRoadsigns2/Eval',
    target_size=(32, 32),
    batch_size=BS,
    class_mode='input'
)

anomaly_generator = test_datagen.flow_from_directory(
    'data/-1',
    target_size=(32, 32),
    batch_size=BS,
    class_mode='input'
)

es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                   patience=5)  # Early stopping (stops training when
# validation doesn't improve for {patience} epochs)
model_filepath = './output/germansignsnetautoenc.4'
save_best = keras.callbacks.ModelCheckpoint(model_filepath, monitor='val_loss',
                                            save_best_only=True, mode='min')

H = autoenc.fit(
    train_generator,
    steps_per_epoch=1000,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=1000,
    shuffle=True, callbacks=[save_best, es])

model = keras.models.load_model(model_filepath)

N = np.arange(0, es.stopped_epoch)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()

print("[INFO] making predictions...")
data_list = []
batch_index = 0
while batch_index <= train_generator.batch_index:
    data = train_generator.next()
    data_list.append(data[0])
    batch_index = batch_index + 1

predicted = model.predict(data_list[0])
no_of_samples = 4
_, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
axs = axs.flatten()
imgs = []
for i in range(no_of_samples):
    imgs.append(data_list[i][i])
    imgs.append(predicted[i])
for img, ax in zip(imgs, axs):
    ax.imshow(img)
plt.show()

print(
    f"Error on validation set:{model.evaluate_generator(validation_generator)}, "
    f"error on anomaly set:{model.evaluate_generator(anomaly_generator)}")
