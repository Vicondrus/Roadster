import argparse
import shutil
import os
import random
import csv

from skimage import transform
from skimage import exposure
from skimage import io

import numpy as np


def makeTop5Stats(name):
    rows = open(name).read().strip().split("\n")

    classes_dict = {}

    for (i, row) in enumerate(rows):

        if i > 0 and i % 100 == 0:
            print("[INFO] processed {} total rows".format(i))

        first, second, third = row.strip().split(",")[:3]

        classId = row.strip().split(",")[-1:]

        id = int(classId[0])

        if str(id) == first or str(id) == second or str(id) == third:
            if id not in classes_dict:
                classes_dict[id] = 1
            else:
                classes_dict[id] += 1

    return classes_dict


def makeLabelCsv(basePath, csvName):
    root_dir = os.path.abspath(basePath)
    writer = csv.writer(open(os.path.sep.join([basePath, csvName+".csv"]), "w"), delimiter=",", lineterminator="\n")
    header = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
    writer.writerow(header)
    for item in os.listdir(root_dir):
        item_full_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_full_path):
            continue
        for image in os.listdir(item_full_path):
            row = ["X", "X", "X", "X", "X", "X", item, item+"/"+image]
            writer.writerow(row)


def split_to_train_and_eval(basePath, csvPath, newPath, evalsize=20):
    # count the number of images
    rows = open(csvPath).read().strip().split("\n")[1:]
    random.shuffle(rows)

    eval_dict = {}

    writerTrain = csv.writer(open(os.path.sep.join([newPath, "Train.csv"]), "w"), delimiter=",", lineterminator="\n")
    writerEval = csv.writer(open(os.path.sep.join([newPath, "Eval.csv"]), "w"), delimiter=",", lineterminator="\n")

    header = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
    writerEval.writerow(header)
    writerTrain.writerow(header)

    # for each image
    for (i, row) in enumerate(rows):

        if i > 0 and i % 1000 == 0:
            print("[INFO] processed {} total images".format(i))

        # find path and id
        (label, imagePath) = row.strip().split(",")[-2:]

        dstTrainPath = os.path.sep.join([newPath, "Train", imagePath.partition("/")[2]])

        dstEvalPath = os.path.sep.join([newPath, "Eval", imagePath.partition("/")[2]])

        imagePath = os.path.sep.join([basePath, imagePath])

        # jpgfile = Image.open(imagePath)

        intaux = int(label)

        if intaux not in eval_dict:
            eval_dict[intaux] = 1
            x = dstEvalPath.partition("/")[0]
            os.makedirs(dstEvalPath.partition("/")[0], exist_ok=True)
            shutil.copy(imagePath, dstEvalPath)
            row = row.replace("Train", "Eval")
            writerEval.writerow(row.strip().split(","))
        elif eval_dict[intaux] < evalsize:
            eval_dict[intaux] += 1
            shutil.copy(imagePath, dstEvalPath)
            row = row.replace("Train", "Eval")
            writerEval.writerow(row.strip().split(","))
        else:
            os.makedirs(dstTrainPath.partition("/")[0], exist_ok=True)
            shutil.copy(imagePath, dstTrainPath)
            writerTrain.writerow(row.strip().split(","))


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

        if i % 10 == 0:
            print("[INFO] Evaluated using {} images".format(i))

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
def load_data_and_labels(basePath, csvPath, evaluation_split=False, evalsize=20):
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
            elif len(eval_dict[intaux]) < evalsize - 1:
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
