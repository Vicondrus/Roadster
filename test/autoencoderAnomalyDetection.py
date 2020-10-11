import os

import cv2
import keras
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator

from trafficSignAutoencoder import TrafficSignNet_Autoencoder

from sklearn.neighbors import KernelDensity

import numpy as np

model = keras.models.load_model("./output/germansignsnetautoenc.3")

basePath = 'D:\\Users\\Victor\\Documents\\GitHub\\Roadster\\data\\germanRoadsigns3'

# evalPath = os.path.sep.join([basePath, "Eval.csv"])
# evalX, _ = load_data_and_labels(basePath, evalPath)
#
# evalX = evalX.astype("float32") / 255.0
#
# predicted = model.predict(evalX)
# no_of_samples = 4
# _, axs = plt.subplots(no_of_samples, 2, figsize=(5, 8))
# axs = axs.flatten()
# imgs = []
# for i in range(no_of_samples):
#     imgs.append(evalX[i] * 255)
#     imgs.append(predicted[i] * 255)
# for img, ax in zip(imgs, axs):
#     ax.imshow(img)
# plt.show()

batch_size = 85
train_datagen = ImageDataGenerator(rescale=1. / 255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory(
    'data/germanRoadsigns2/Train',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='input'
)

test_datagen = ImageDataGenerator(rescale=1. / 255, data_format='channels_last')
validation_generator = test_datagen.flow_from_directory(
    'data/germanRoadsigns2/Eval',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='input'
)

anomaly_generator = test_datagen.flow_from_directory(
    'data/-1',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='input'
)

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

print(model.summary())

encoder = model.get_layer(name="encoder")

print(encoder.summary())

encoded_images = encoder.predict_generator(train_generator)

validation_encoded = encoder.predict_generator(validation_generator)

anom_encoded = encoder.predict_generator(anomaly_generator)

kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(encoded_images)
training_density_scores = kde.score_samples(encoded_images)
validation_density_scores = kde.score_samples(validation_encoded)
anomaly_density_scores = kde.score_samples(anom_encoded)

plt.figure(figsize=(10, 7))
plt.title('Distribution of Density Scores')
# plt.hist(training_density_scores, 20, alpha=0.5, label='Training Normal',
#          range=(-50, 5))
plt.hist(validation_density_scores, 20, alpha=0.5, label='Validation Normal',
         range=(-50, 5))
plt.hist(anomaly_density_scores, 20, alpha=0.5, label='Anomalies', range=(-50, 5))
plt.legend(loc='upper right')
plt.xlabel('Density Score')

plt.show()
