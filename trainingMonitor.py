from functools import singledispatch

from keras.callbacks import BaseLogger
import matplotlib.pyplot as plot
import numpy as np
import json
import os
import cv2
import glob


@singledispatch
def to_serializable(self):
    """Used by default."""
    return str(self)


@to_serializable.register(np.float32)
def ts_float32(self):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(self)


class TrainingMonitor(BaseLogger):

    def __init__(self, figPath, jsonPath=None, startAt=0):
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs={}):
        self.H = {}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H, default=to_serializable))
            f.close()

        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plot.style.use("ggplot")
            plot.figure()
            plot.plot(N, self.H["loss"], label="train_loss")
            plot.plot(N, self.H["val_loss"], label="val_loss")
            plot.plot(N, self.H["accuracy"], label="train_acc")
            plot.plot(N, self.H["val_accuracy"], label="val_acc")
            plot.title("Training Loss and Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plot.xlabel("Epoch #")
            plot.ylabel("Loss/Accuracy")
            plot.legend()

            plot.savefig(self.figPath)
            plot.close()

            data_path = os.path.join(self.figPath)
            files = glob.glob(data_path)
            for f in files:
                image = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                cv2.imshow("plot", image)
                cv2.waitKey(1)
