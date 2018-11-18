import cv2
import tensorflow as tf

# will use this to convert prediction num to string value
CATEGORIES = ['apple_pie', 'beignets', 'bibimbap', 'breakfast_burrito',
              'cheese_plate', 'chicken_wings', 'creme_brulee', 'deviled_eggs',
              'dumplings', 'lobster_bisque', 'croque_madame', 'shrimp_and_grits',
              'guacamole', 'tuna_tartare', 'peking_duck', 'macarons',
              'paella', 'strawberry_shortcake', 'ramen', 'red_velvet_cake', 'samosa',
              'cannoli', 'ceviche', 'baby_back_ribs']

img = 'redVelvet.png'
IMG_SIZE = 50


def prepare(filepath):
    img_array = cv2.imread(filepath)  # read in the image
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)  # return the image with shaping that TF wants.


model = tf.keras.models.load_model("food_prediction.model")
prediction = model.predict([prepare(img)])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

print(prediction)
