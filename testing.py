import collections

import matplotlib.pyplot as plot
import tensorflow as tf
from sklearn.metrics import classification_report

import util


def plotStats():
    statsPre = util.makeTop5Stats(".\\output\\germansignsnet1.8\\top5.csv")

    stats = collections.OrderedDict()

    for key in sorted(statsPre):
        stats[key] = statsPre[key]

    plot.bar(range(len(stats)), list(stats.values()), align='center')
    plot.xticks(range(len(stats)), list(stats.keys()), rotation=90)
    plot.savefig(".\\output\\germansignsnet1.8\\classstats.jpg")
    plot.close()


def evalModel():
    model = tf.keras.models.load_model(".\\output\\germansignsnet1.8")

    evalX, evalY = util.load_data_and_labels(".\\data\\germanRoadsigns2", ".\\data\\germanRoadsigns2\\Eval.csv")

    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    stats, top5, report, confusion = util.evaluate(model, evalX, evalY, labelNames)

    f = open("output/germansignsnet1.8/classification_report.txt", "w+")
    f.write(report)
    f.close()

    tf.print(confusion)

    util.writeTopToCSV('.\\output\\germansignsnet1.8\\top5.csv', top5)

    plot.bar(range(len(stats)), list(stats.values()), align='center')
    plot.xticks(range(len(stats)), list(stats.keys()))
    plot.savefig(".\\output\\germansignsnet1.8\\stats.jpg")
    plot.close()


evalModel()
plotStats()
