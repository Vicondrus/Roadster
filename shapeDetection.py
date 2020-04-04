import cv2
import imutils
import numpy as np
from math import sqrt
from tensorflow.keras.models import load_model
from skimage import exposure, io
from skimage import transform

model = load_model(".\\output\\germansignsnet3.5")
labelNames = open("signnames.csv").read().strip().split("\n")[1:]
labelNames = [l.split(",")[1] for l in labelNames]

label = None
prevAcc = None

i = 0


# save images extracted from video
# label them
# evaluate classifier

# implement testing

# make evaluation after loading model - method DONE

# separate BY PYTHON SCRIPT the evaluation from training DONE

# make statistics on top5 evaluation - check how many were classified correctly on top3 DONE

def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(img_hist_equalized)
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)

    cv2.imshow("Contrast", img_hist_equalized)
    # cv2.waitKey()
    return img_hist_equalized


def filterColors(image):
    img_filtered = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 70])
    upper_red = np.array([10, 255, 255])

    mask1 = cv2.inRange(img_filtered, lower_red, upper_red)

    lower_red = np.array([170, 100, 70])
    upper_red = np.array([180, 255, 255])

    mask2 = cv2.inRange(img_filtered, lower_red, upper_red)

    sens = 3
    lower_white = np.array([0, 0, 255 - sens])
    upper_white = np.array([255, sens, 255])

    mask3 = cv2.inRange(img_filtered, lower_white, upper_white)

    low_blue = np.array([100, 150, 0])
    high_blue = np.array([140, 255, 255])

    mask4 = cv2.inRange(img_filtered, low_blue, high_blue)

    low_yellow = np.array([10, 100, 70])
    high_yellow = np.array([30, 255, 255])

    mask5 = cv2.inRange(img_filtered, low_yellow, high_yellow)

    img_filtered = mask1 + mask2 + mask3 + mask4 + mask5
    output_img = image.copy()
    output_img[np.where(img_filtered == 0)] = 0

    cv2.imshow("masked", output_img)

    return output_img


def auto_canny(image, sigma=0.33):
    v = np.median(image)

    image = cv2.GaussianBlur(image, (3, 3), 0)

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    cv2.imshow("Canny edge", edged)

    return edged


def laplacianOfGaussian(image):
    LoG = cv2.GaussianBlur(image, (3, 3), 0)  # paramter
    gray = cv2.cvtColor(LoG, cv2.COLOR_BGR2GRAY)
    LoG = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)  # parameter
    LoG = cv2.convertScaleAbs(LoG)
    cv2.imshow("Laplacian of Gaussian", LoG)
    # cv2.waitKey()
    return LoG


def binarization(image):
    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Binarized", image)
    # cv2.waitKey()
    # thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return thresh


def preprocess_image(image):
    image = constrastLimit(image)
    image = filterColors(image)
    image = auto_canny(image)
    image = binarization(image)
    return image


# Find Signs
def removeSmallComponents(image, threshold):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    img2 = np.zeros((output.shape), dtype=np.uint8)
    # for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def findContour(image):
    # find contours in the threshed image
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    return cnts


def contourIsSign(perimeter, centroid, threshold):
    #  perimeter, centroid, threshold
    # # Compute signature of contour
    result = []
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]

    # Check signature of contour.
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)

    if temp < threshold:  # is  the sign
        return True, max_value + 2
    else:  # is not the sign

        return False, max_value + 2


# crop sign
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height - 1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance + 1), width - 1])
    print(left, right, top, bottom)
    return image[left:right, top:bottom]


def cropSign(image, coordinate, diff=10):
    width = image.shape[1]
    height = image.shape[0]
    if height > width:
        width = height
    else:
        height = width
    top = max([int(coordinate[0][1]) - diff, 0])
    bottom = min([int(coordinate[1][1]) + diff, height - 1])
    left = max([int(coordinate[0][0]) - diff, 0])
    right = min([int(coordinate[1][0]) + diff, width - 1])
    # print(top,left,bottom,right)
    return image[top:bottom, left:right]


def findLargestSign(image, contours, threshold, distance_theshold):
    global prevAcc, label
    global i
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1 - threshold)
        if is_sign and distance > max_distance and distance > distance_theshold:
            contour = image.copy()
            cv2.drawContours(contour, c, -1, (0, 255, 0), 2)
            cv2.circle(contour, (cX, cY), 5, (255, 0, 0), 2)
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = cropSign(image, coordinate)
            # sign = constrastLimit(sign)
            cv2.imshow("sign", sign)

            cv2.imwrite(".\\data\\videoImages\\x.jpg", sign)
            i += 1

            cv2.imshow("contour", contour)
            cv2.waitKey(1)
            obj = io.imread(".\\data\\videoImages\\x.jpg")
            obj = transform.resize(obj, (32, 32))
            obj = exposure.equalize_adapthist(obj, clip_limit=0.1)

            cv2.imshow("again", obj)

            obj = obj.astype("float32") / 255.0
            obj = np.expand_dims(obj, axis=0)

            # print(preds.max(axis=1)[0], labelNames[preds.argmax(axis=1)[0]])
            preds = model.predict(obj)
            top = np.argsort(-preds, axis=1)
            print(preds[0][top[0][0]], labelNames[top[0][0]])
            print(preds[0][top[0][1]], labelNames[top[0][1]])
            print(preds[0][top[0][2]], labelNames[top[0][2]])
            j = top[0][0]
            if preds.max(axis=1)[0] > 0.9:
                 if prevAcc is None or preds.max(axis=1)[0] > 0.999:
                    prevAcc = preds.max(axis=1)[0]
                    label = labelNames[j]
    return sign, coordinate, label


def localization(image, min_size_components, similitary_contour_with_circle):
    original_image = image.copy()
    binary_image = preprocess_image(image)

    label = None

    binary_image = removeSmallComponents(binary_image, min_size_components)
    cv2.imshow('BINARY IMAGE', binary_image)
    cv2.waitKey(1)
    contours = findContour(binary_image)
    # signs, coordinates = findSigns(image, contours, similitary_contour_with_circle, 15)
    if contours is not None:
        sign, coordinate, label = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    else:
        coordinate = [(0, 0), (0, 0)]

    return coordinate, original_image, label


vidcap = cv2.VideoCapture('video/video2.avi')

while True:
    success, frame = vidcap.read()
    if success is False:
        break
    coordinate, image, label = localization(frame, 300, 0.65)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.putText(frame, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 255), 2)
    cv2.imshow("Video", frame)

vidcap.release()
cv2.destroyAllWindows()
