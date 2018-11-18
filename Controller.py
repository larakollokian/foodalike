import numpy as np
from flask import Flask, abort, jsonify, request, app
import tensorflow as tf
import pickle
import base64
import cv2

# model = tf.keras.models.load_model("food_prediction.model")
model = pickle.load(open("food_classifier.pickle", "rb"))
IMG_SIZE = 50

app = Flask(__name__)


def prepare(image):
    img_array = cv2.imread(image)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.


@app.route("/")
def hello():
    return "Hello World"


@app.route('/api', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)

    predict_request = base64.b64decode(data)
    # decoded_request = predict_request.decode()
    # predict_request = np.array[predict_request]

    prediction = model.predict([prepare(predict_request)])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT



    # y_hat = model(predict_request)
    output = [y_hat[0]]

    return jsonify(results=output)


if __name__ == '__main__':
    app.run("https://foodalike4.herokuapp.com/")
    # app.run(port=9000, debug=True)
