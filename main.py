import cv2
from tensorflow.keras.models import load_model

import classicalShapeDetection
import voting


def main():
    models = []
    model = load_model(".\\output\\germansignsnet3.7")
    models.append(model)
    model = load_model(".\\output\\germansignsnet1.8")
    models.append(model)
    model = load_model(".\\output\\germansignsnet4.4")
    models.append(model)
    labelNames = open("signnames.csv").read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    vidcap = cv2.VideoCapture('video/video5.mp4')

    label = None

    while True:
        success, frame = vidcap.read()
        if success is False:
            break
        coordinate, image, sign = classicalShapeDetection.localization(frame, 300, 0.65)
        if sign is not None:
            # yoloShapeDetection.recognizeObjects(image, confidence_thresh=0.4)
            j = voting.vote_on_image(models, sign)
            if j is not None:
                label = labelNames[j]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 255), 2)

        cv2.imshow("Video", frame)

    vidcap.release()
    cv2.destroyAllWindows()
    classicalShapeDetection.end()


main()
