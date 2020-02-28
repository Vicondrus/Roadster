import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to pre-trained traffic sign recognizer")
args = vars(ap.parse_args())

print("[INFO] loading model...")
model = load_model(args["model"])

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)