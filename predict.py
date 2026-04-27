import tensorflow as tf
import numpy as np
import cv2

# Load trained model
model = tf.keras.models.load_model("waste_classifier.h5")

# Class labels (MUST match folder names)
labels = ['glass', 'metal', 'organic', 'paper', 'plastic']

waste_info = {
    "glass": "Glass is 100% ecyclable and can be reused indefinitely without losing quality.",
    "metal": "Metal can be recycled multiple times and should be sent to scrap or recycling centers.",
    "organic": "Organic waste decomposes naturally and can be composted into nutrient-rich fertilizer.",
    "paper": "Paper is recyclable and can be reused to make new paper products.",
    "plastic": "Plastic can be recycled but takes hundreds of years to decompose naturally."
}

# Load and preprocess image
img = cv2.imread("test.jpg")

if img is None:
    print("Error: Image not found!")
    exit()

img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
class_id = np.argmax(prediction)
confidence = prediction[0][class_id] * 100
info = waste_info[prediction]

print("Predicted Waste Type:", labels[class_id])
print("Confidence:", round(confidence, 2), "%")
