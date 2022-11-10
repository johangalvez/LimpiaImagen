import matplotlib.pyplot as plt
import keras_ocr

pipeline = keras_ocr.pipeline.Pipeline()
#read image from the an image path (a jpg/png file or an image url)
img = keras_ocr.tools.read("img/foto.png")
# Prediction_groups is a list of (word, box) tuples
prediction_groups = pipeline.recognize([img])
#print image with annotation and boxes
keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])
#for text, box in prediction_groups[0]:
#    print(text)  

# Plot the predictions
fig, axs = plt.subplots(nrows=len(img), figsize=(20, 20))
for ax, image, predictions in zip(axs, img, prediction_groups):
    keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)


