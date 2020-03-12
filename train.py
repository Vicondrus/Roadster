import matplotlib

from trafficSignCnn import TrafficSignNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io

import matplotlib.pyplot as plot

import numpy as np
import argparse
import random
import os

from keras.callbacks import ModelCheckpoint
from trainingMonitor import TrainingMonitor

matplotlib.use("Agg")


def load_data_and_labels(basePath, csvPath):
    data = []
    labels = []

    # count the number of images
    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    # for each image
    for (i, row) in enumerate(rows):

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        # find path and id
        (label, imagePath) = row.strip().split(",")[-2:]

        imagePath = os.path.sep.join([basePath, imagePath])
        image = io.imread(imagePath)

        # resize and
        image = transform.resize(image, (32, 32))

        # Contrast Limited Adaptive Histogram Equalization (CLAHE).
        #
        # An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of
        # the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of
        # the image.
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        # append the transformed image to the data array
        data.append(image)
        # and the given label too
        labels.append(int(label))

    data = np.array(data)
    labels = np.array(labels)

    return data, labels


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input training model")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output dictionary")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to training history plot")
args = vars(ap.parse_args())

# default parameters, might need adjustment
NUM_EPOCHS = 30
INIT_LR = 1e-3
BS = 64

labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])

# load data to a tuple with the previous function
print("[INFO] loading training and testing data...")
# train data
(trainX, trainY) = load_data_and_labels(args["dataset"], trainPath)
# test data
(testX, testY) = load_data_and_labels(args["dataset"], testPath)

# scale to [0, 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert to one-hot encoding (boolean values from which only one is true at a time)
numLabels = len(np.unique(trainY))
trainY = to_categorical(trainY, numLabels)
testY = to_categorical(testY, numLabels)

classTotals = trainY.sum(axis=0)
classWeight = classTotals.max() / classTotals

aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest"
)

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet.build(width=32, height=32, depth=3, classes=numLabels)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# training monitor
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0] // BS,
    epochs=NUM_EPOCHS,
    class_weight=classWeight,
    callbacks=callbacks,
    verbose=1
)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))

print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

N = np.arange(0, NUM_EPOCHS)
plot.style.use("ggplot")
plot.figure()
plot.plot(N, H.history["loss"], label="train_loss")
plot.plot(N, H.history["val_loss"], label="val_loss")
plot.plot(N, H.history["accuracy"], label="train_acc")
plot.plot(N, H.history["val_accuracy"], label="val_acc")
plot.title("Training Loss and Accuracy on Dataset")
plot.xlabel("Epoch #")
plot.ylabel("Loss/Accuracy")
plot.legend(loc="lower left")
plot.savefig(args["plot"])
