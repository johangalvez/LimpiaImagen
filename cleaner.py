#https://www.youtube.com/watch?v=3RNPJbUHZKs
#https://towardsdatascience.com/remove-text-from-images-using-cv2-and-keras-ocr-24e7612ae4f4

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np 

def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

pipeline = keras_ocr.pipeline.Pipeline()

def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        #img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        cv2.imshow('image',img)
        cv2.waitKey(0)
                 
    return(img)

img2=inpaint_text("img/foto.png", pipeline)
cv2.imshow('image',img2)
cv2.waitKey(0)



