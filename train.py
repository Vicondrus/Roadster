import argparse
import csv
import io as io_simple
import os
import random

import keras
import matplotlib
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from skimage import exposure
from skimage import io
from skimage import transform
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from trafficSignCnn_v4 import TrafficSignNet_v4
from trainingMonitor import TrainingMonitor

matplotlib.use("Agg")

EVALSIZE = 20

# data split and training done according to Adrian Rosebrock's model, found at https://www.pyimagesearch.com/2019/11/04/traffic-sign-classification-with-keras-and-deep-learning/

def writeTopToCSV(name, list):
    with open(name, mode='w') as top_file:
        top_writer = csv.writer(top_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC,
                                lineterminator="\n")

        for top in list:
            top_writer.writerow(top)


def evaluate(model, evalX, evalY):
    stats = {0: 0, 1: 0, 2: 0, 3: 0}
    top5 = []
    for i, image in enumerate(evalX):
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        preds = model.predict(image)
        top = np.argsort(-preds, axis=1)
        if evalY[i] == top[0][0]:
            stats[0] += 1
        elif evalY[i] == top[0][1]:
            stats[1] += 1
        elif evalY[i] == top[0][2]:
            stats[2] += 1
        else:
            stats[3] += 1
        top5.append([top[0][0], top[0][1], top[0][2], top[0][3], top[0][4], evalY[i]])
    return stats, top5


# split training data again into TRAINING and FINAL EVALUATION/TESTING - DISJOINT
def load_data_and_labels(basePath, csvPath, evaluation_split=False):
    data = []
    labels = []

    # count the number of images
    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    if evaluation_split:
        eval_dict = {}

        # for each image
    for (i, row) in enumerate(rows):

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        # find path and id
        (label, imagePath) = row.strip().split(",")[-2:]

        imagePath = os.path.sep.join([basePath, imagePath])
        image = io.imread(imagePath)

        image = transform.resize(image, (32, 32))

        image = exposure.equalize_adapthist(image, clip_limit=0.1)

        intaux = int(label)

        if evaluation_split:
            if intaux not in eval_dict:
                eval_dict[intaux] = [image]
            elif len(eval_dict[intaux]) < EVALSIZE - 1:
                eval_dict[intaux].append(image)
            else:
                data.append(image)
                labels.append(intaux)
        else:
            data.append(image)
            labels.append(intaux)

    if evaluation_split:
        eval_data = []
        eval_labels = []
        for key in eval_dict:
            for value in eval_dict[key]:
                eval_data.append(value)
                eval_labels.append(key)

    data = np.array(data)
    labels = np.array(labels)

    if evaluation_split:
        return ((eval_data, eval_labels), (data, labels))

    return data, labels


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input training model")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-o", "--output", required=True, help="path to output dictionary")
ap.add_argument("-p", "--plot", type=str, default="plot.png", help="path to training history plot")
args = vars(ap.parse_args())

# default parameters, might need adjustment
NUM_EPOCHS = 100
INIT_LR = 1e-3
BS = 64

labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

trainPath = os.path.sep.join([args["dataset"], "Train.csv"])
testPath = os.path.sep.join([args["dataset"], "Test.csv"])

# load data to a tuple with the previous function
print("[INFO] loading training, validation and evaluation data...")
# train data
((evalX, evalY), (trainX, trainY)) = load_data_and_labels(args["dataset"], trainPath, evaluation_split=True)
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
base_learning_rate = 0.0001

opt = Adam(lr=INIT_LR, decay=INIT_LR / (NUM_EPOCHS * 0.5))
model = TrafficSignNet_v4.build(width=32, height=32, depth=3, classes=numLabels)

file_writer = tf.summary.create_file_writer("logs\\fit\\" + args["output"] + "\\cm")


def log_confusion_matrix(epoch, logs):
    # Use the model to predict the values from the validation dataset.
    test_pred = model.predict(testX)

    con_mat = tf.math.confusion_matrix(labels=testY.argmax(axis=1), predictions=test_pred.argmax(axis=1)).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    classes = [i for i in range(numLabels)]

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)

    figure = plot.figure(figsize=(16, 16))
    sns.heatmap(con_mat_df, annot=True, cmap=plot.cm.Blues)
    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')

    buf = io_simple.BytesIO()
    plot.savefig(buf, format='png')

    plot.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)

    # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", image, step=epoch)


cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

log_dir = "logs\\fit\\" + args["output"]
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# training monitor
figPath = os.path.sep.join([args["output"], "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath),
             tensorboard_callback, cm_callback, EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)]

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

print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"])

evalXDum = list(evalX)
evalYDum = list(evalY)

stats, top5 = evaluate(model, evalXDum, evalYDum)

writeTopToCSV(args["model"] + '\\top5.csv', top5)

plot.bar(range(len(stats)), list(stats.values()), align='center')
plot.xticks(range(len(stats)), list(stats.keys()))
plot.savefig(args["model"] + "\\stats.jpg")

evalX = np.array(evalX, dtype=np.float32) / 255.0
evalY = to_categorical(evalY, numLabels)

print("[INFO] evaluating network...")
predictions = model.predict(evalX, batch_size=BS)
f = open(args["model"] + "/classification_report.txt", "w+")
f.write(classification_report(evalY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
f.close()

# add evaluation part
# for each model find how many were predicted right first try
#                                                       second try
#                                                       third try

# for each prediction print top 5 and what it should have been

N = np.arange(0, len(H["loss"]))
plot.style.use("ggplot")
plot.figure()
plot.plot(N, H["loss"], label="train_loss")
plot.plot(N, H["val_loss"], label="val_loss")
plot.plot(N, H["accuracy"], label="train_acc")
plot.plot(N, H["val_accuracy"], label="val_acc")
plot.title("Training Loss and Accuracy [Epoch {}]".format(
    len(H["loss"])))
plot.xlabel("Epoch #")
plot.ylabel("Loss/Accuracy")
plot.legend(loc="lower left")
plot.savefig(args["plot"])
