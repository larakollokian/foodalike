import os
import cv2
import random
import numpy as np
import pickle

DATADIR = "C:/Users/Lara/workspaceF18/ImageRecognition/food-101/images"

CATEGORIES = ['apple_pie', 'beignets', 'bibimbap', 'breakfast_burrito',
              'cheese_plate', 'chicken_wings', 'creme_brulee', 'deviled_eggs',
              'dumplings', 'lobster_bisque', 'croque_madame', 'shrimp_and_grits',
              'guacamole', 'tuna_tartare', 'peking_duck', 'macarons',
              'paella', 'strawberry_shortcake', 'ramen', 'red_velvet_cake', 'samosa',
              'cannoli', 'ceviche', 'baby_back_ribs']

IMG_SIZE = 50
training_data = []


def create_training_data():
    for category in CATEGORIES:
        print(category)
        path = os.path.join(DATADIR, category)  # joins path of directory to each category
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):  # iterate over each image in each category
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])


create_training_data()
print(len(training_data))

random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
