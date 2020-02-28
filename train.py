import matplotlib
matplotlib.use("Agg")

from TrafficSignNet import TrafficSignNet
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

def load_split(basePath, csvPath):
    data=[]
    labels=[]

    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    for (i, row) in enumerate(rows):
        if i>0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        (label, imagePath) = row.strip().split(",")[-2:]

        imagePath = os.path.sep.join([basePath,imagePath])
        image = io.imread(imagePath)

        image = transform.resize(image, (32,32))
        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        data.append(image)
        labels.append(int(label))

    data = np.array(data)
    label = np.array(labels)

    return (data, labels)

ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="path to input training model")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to training history plot")
args=vars(ap.parse_args())

NUM_EPOCHS=30
INIT_LR=1e-3
BS=64

labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

trainPath = os.path.sep.join([args["dataset"],"Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])

print("[INFO] loading training and testing data...")
(trainX, trainY) =load_split(args["dataset"], trainPath)
(testX, testY) = load_split(args["dataset"], testPath)

trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

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

print("[INFO] training network...")
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=trainX.shape[0]//BS,
    epochs=NUM_EPOCHS,
    class_weight=classWeight,
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