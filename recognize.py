__author__ = 'saideeptalari'
import numpy as np
import cv2
import imutils
import argparse
from skimage.filters import threshold_adaptive
from keras.models import load_model

#parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="Path to image to recognize")
ap.add_argument("-m","--model",required=True,help="Path to saved classifier")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image,width=320)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
_,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
thresh = cv2.dilate(thresh,None)

(cnts,_) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])

digits = []
boxes = []

for i,c in enumerate(cnts):
    if cv2.contourArea(c)<avgCntArea/10:
        continue
    mask = np.zeros(gray.shape,dtype="uint8")

    (x,y,w,h) = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    cv2.drawContours(mask,[hull],-1,255,-1)
    mask = cv2.bitwise_and(thresh,thresh,mask=mask)

    digit = mask[y-8:y+h+8,x-8:x+w+8]
    digit = cv2.resize(digit,(28,28))
    boxes.append((x,y,w,h))
    digits.append(digit)

digits = np.array(digits)
model = load_model(args["model"])
#digits = digits.reshape(-1,784)    #for mlp
digits = digits.reshape(digits.shape[0],28,28,1)    #for cnn
labels = model.predict_classes(digits)

cv2.imshow("Original",image)
cv2.imshow("Thresh",thresh)

for (x,y,w,h),label in sorted(zip(boxes,labels)):
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
    cv2.putText(image,str(label),(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.imshow("Recognized",image)
    cv2.waitKey(0)

cv2.destroyAllWindows()