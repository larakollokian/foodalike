import cv2
import requests
import json
import base64
import pickle
from flask import jsonify
import keras as K
import numpy as np
import tensorflow as tf

url = "http://localhost:9000/api"

img = "C:/Users/Lara/workspaceF18/ImageRecognition/food-101/images/dumplings/432.jpg"

food = pickle.load(open("food_classifier.pickle", "rb"))

IMG_SIZE = 50

img_array = cv2.imread(img)
new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

"""listified_array = new_array.tolist()
data = json.dumps(listified_array)

enc = data.encode()

encoded = base64.b64encode(enc)

# print(encoded)

r = requests.post(url, encoded)

# print(r.json())
"""

tensor_array = tf.convert_to_tensor(new_array)
y_hat = food(tensor_array)
output = [y_hat[0]]

print(jsonify(results=output))
