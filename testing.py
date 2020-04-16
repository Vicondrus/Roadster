from trafficSignCnn_v3 import TrafficSignNet_v3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from skimage import transform
from skimage import exposure
from skimage import io

import matplotlib.pyplot as plot

import tensorflow as tf
import collections

import util

import numpy as np
import random
import os

from trainingMonitor import TrainingMonitor


def plotStats():
    statsPre = util.makeTop5Stats(".\\output\\germansignsnet4.1\\top5.csv")

    stats = collections.OrderedDict()

    for key in sorted(statsPre):
        stats[key] = statsPre[key]

    plot.bar(range(len(stats)), list(stats.values()), align='center')
    plot.xticks(range(len(stats)), list(stats.keys()), rotation=90)
    plot.savefig(".\\output\\germansignsnet4.1\\classstats.jpg")


def evalModel():
    model = tf.keras.models.load_model(".\\output\\germansignsnet4.1")

    evalX, evalY = util.load_data_and_labels(".\\data\\germanRoadsigns2", ".\\data\\germanRoadsigns2\\Eval.csv")

    stats, top5 = util.evaluate(model, evalX, evalY)

    evalX = list(evalX)
    evalY = list(evalY)

    util.writeTopToCSV('.\\output\\germansignsnet4.1\\top5.csv', top5)

    plot.bar(range(len(stats)), list(stats.values()), align='center')
    plot.xticks(range(len(stats)), list(stats.keys()))
    plot.savefig(".\\output\\germansignsnet4.1\\stats.jpg")

plotStats()