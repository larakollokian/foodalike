import numpy as np
from flask import Flask, abort, jsonify, request, app
import pickle
import base64

my_random_forest = pickle.load(open("food_classifier.pickle", "rb"))

app = Flask(__name__)


@app.route('/api', methods=['POST'])
def make_predict():
    data = request.get_json(force=True)

    predict_request = base64.b64decode(data)
    decoded_request = predict_request.decode()
    predict_request = np.array[decoded_request]

    y_hat = my_random_forest(predict_request)
    output = [y_hat[0]]

    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=9000, debug=True)
