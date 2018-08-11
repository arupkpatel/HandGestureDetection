from keras.models import load_model

classifier = load_model('cdi.h5')

import numpy as np
from keras.preprocessing import image
import cv2
import os

img = cv2.imread('spred\\9.jpg',0)
img2 =cv2.imread('spred\\8.jpg')
imgbw = cv2.resize(img, (48, 48))
imgbw = image.img_to_array(imgbw)
imgbw = np.expand_dims(imgbw, axis=0)
result = classifier.predict(imgbw)
# print(result)

if result[0][0] == 0:
    print('1')
    os.system('notepad')
    cv2.putText(img2, 'Gesture Recognised: Notepad', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0),
                thickness=3)
else:
    print('2')
    os.system('calc')
    cv2.putText(img2, 'Gesture Recognised: Calculator', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=3)


cv2.imshow('Output',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()