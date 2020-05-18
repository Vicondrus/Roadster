import collections

import matplotlib.pyplot as plot
import tensorflow as tf

import util


def plotStats():
    statsPre = util.makeTop5Stats(".\\output\\germansignsnet4.4\\top5.csv")

    stats = collections.OrderedDict()

    for key in sorted(statsPre):
        stats[key] = statsPre[key]

    plot.bar(range(len(stats)), list(stats.values()), align='center')
    plot.xticks(range(len(stats)), list(stats.keys()), rotation=90)
    plot.savefig(".\\output\\germansignsnet4.4\\classstats.jpg")
    plot.close()


def evalModel():
    model = tf.keras.models.load_model(".\\output\\germansignsnet4.4")

    evalX, evalY = util.load_data_and_labels(".\\data\\germanRoadsigns2", ".\\data\\germanRoadsigns2\\Eval.csv")

    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    stats, top5, report, confusion = util.evaluate(model, evalX, evalY, labelNames)

    f = open("output/germansignsnet4.4/classification_report.txt", "w+")
    f.write(report)
    f.close()

    tf.print(confusion)

    util.writeTopToCSV('.\\output\\germansignsnet4.4\\top5.csv', top5)

    plot.bar(range(len(stats)), list(stats.values()), align='center')
    plot.xticks(range(len(stats)), list(stats.keys()))
    plot.savefig(".\\output\\germansignsnet4.4\\stats.jpg")
    plot.close()


evalModel()
plotStats()
