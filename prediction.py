import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os


def skinseg(img):
    # img = cv2.imread(imgstr)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype='uint8')
    upper = np.array([20, 255, 255], dtype='uint8')
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    op = cv2.bitwise_and(img, img, mask=mask)
    return op


def gesturerecog(imgpara):
    img = cv2.cvtColor(imgpara,cv2.COLOR_BGR2GRAY)
    # img2 = imgpara
    imgbw = cv2.resize(img, (48, 48))
    imgbw = image.img_to_array(imgbw)
    imgbw = np.expand_dims(imgbw, axis=0)
    result = classifier.predict(imgbw)
    return result


classifier = load_model('cdi.h5')
cam = cv2.VideoCapture(0)
while True:
    _, rtimg = cam.read()
    testimg = skinseg(rtimg)

    testresult = gesturerecog(testimg)

    if testresult[0][0] == 0:
        print('1')
        # os.system('notepad')
        cv2.putText(testimg, 'Gesture Recognised: Notepad', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0),
                    thickness=3)
    else:
        print('2')
        # os.system('calc')
        cv2.putText(testimg, 'Gesture Recognised: Calculator', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 0), thickness=3)

    cv2.imshow('hand', testimg)

    k = cv2.waitKey(30)
    if k==27:
        break



cv2.destroyAllWindows()
