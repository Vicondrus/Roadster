from skimage.measure import compare_ssim as ssim
import os
import cv2 as cv
import glob

list_xml_files = glob.glob('haarstages/*.xml')
video_capture = cv.VideoCapture('video/videoplayback.mp4')

while True:
    red, image = video_capture.read()

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for filename in list_xml_files:
        # print(filename)
        sign_cascade = cv.CascadeClassifier(filename)
        signs = sign_cascade.detectMultiScale(
            gray,
            scaleFactor=1.4,
            minNeighbors=3
        )
        if len(signs) != 0:
            for (x, y, w, h) in signs:
                obj = gray[y:y + h, x:x + w]
                cv.imshow("Extracted", obj)
                cv.waitKey(1)
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    cv.imshow("Video", image)

video_capture.release()
cv.destroyAllWindows()