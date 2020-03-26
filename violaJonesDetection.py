import numpy as np
from skimage import exposure
from tensorflow.keras.models import load_model
import cv2 as cv
import glob
from skimage import transform

list_xml_files = glob.glob('haarstages/*.xml')
video_capture = cv.VideoCapture('video/video5.mp4')

model = load_model(".\\output\\germansignsnet")
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

while True:
    success, image = video_capture.read()

    if not success:
        break

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    for filename in list_xml_files:

        sign_cascade = cv.CascadeClassifier(filename)
        signs = sign_cascade.detectMultiScale(
            gray,
            scaleFactor=1.4,
            minNeighbors=3
        )
        if len(signs) != 0:
            for (x, y, w, h) in signs:
                org = image[max(0, y - int(11*h/10)):y + int(11*h/10), x:x + w]
                obj = transform.resize(org, (32, 32))
                obj = exposure.equalize_adapthist(obj, clip_limit=0.1)

                obj = obj.astype("float32") / 255.0
                obj = np.expand_dims(obj, axis=0)

                preds = model.predict(obj)
                print(preds.max(axis=1), preds.argmax(axis=1))
                if preds.max(axis=1)[0] < 0.9:
                    continue
                cv.waitKey(1)
                j = preds.argmax(axis=1)[0]
                label = labelNames[j]

                cv.putText(org, label, (5, 15), cv.FONT_HERSHEY_SIMPLEX,
                           0.5, (255, 0, 255), 2)

                cv.imshow("Extracted", org)
                cv.waitKey(1)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    cv.imshow("Video", image)

video_capture.release()
cv.destroyAllWindows()
