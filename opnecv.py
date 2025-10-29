!pip install --quiet opencv-python-headless matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt
from google.colab import files
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

def show_img_bgr(img_bgr, title="Result", figsize=(10,8)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.axis('off')
    plt.imshow(img_rgb)
    plt.title(title)
    plt.show()

def draw_label(img, text, org, bgcolor=(0,0,0), color=(255,255,255), font_scale=0.6, thickness=1):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y), bgcolor, -1)
    cv2.putText(img, text, (x+2, y-4), font, font_scale, color, thickness, cv2.LINE_AA)

uploaded = files.upload()
if len(uploaded) == 0:
    raise RuntimeError("No files uploaded")
fname = next(iter(uploaded.keys()))
img = cv2.imdecode(np.frombuffer(uploaded[fname], np.uint8), cv2.IMREAD_COLOR)
show_img_bgr(img, title="Original image")

model = mobilenet_v2.MobileNetV2(weights='imagenet')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (224, 224))
x = keras_image.img_to_array(resized)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)
decoded = decode_predictions(preds, top=3)[0]
best_label = decoded[0][1]
best_prob = decoded[0][2]
annot = img.copy()
draw_label(annot, f"{best_label} ({best_prob:.2f})", (10, 30), bgcolor=(0,0,0), color=(255,255,255))
show_img_bgr(annot, title=f"Recognized: {best_label}")

print(f"Recognized: {best_label} ({best_prob:.2f})")