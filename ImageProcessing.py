import os
import cv2
import random
import numpy as np
import pickle
import json

DATADIR = "C:/Users/Lara/workspaceF18/ImageRecognition/food-101/images"
TRAIN = "C:/Users/Lara/workspaceF18/ImageRecognition/food-101/meta/train.json"  # define train data
TEST = "C:/Users/Lara/workspaceF18/ImageRecognition/food-101/meta/test.json"  # define test data

CATEGORIES = ['apple_pie', 'beignets', 'bibimbap', 'breakfast_burrito',
              'cheese_plate', 'chicken_wings', 'creme_brulee', 'deviled_eggs',
              'dumplings', 'lobster_bisque', 'croque_madame', 'shrimp_and_grits',
              'guacamole', 'tuna_tartare', 'peking_duck', 'macarons',
              'paella', 'strawberry_shortcake', 'ramen', 'red_velvet_cake', 'samosa',
              'cannoli', 'ceviche', 'baby_back_ribs']

IMG_SIZE = 50

training_data = []
test_data = []

# load json files for data lists
with open(TRAIN, encoding='utf-8') as data_file:
    train_data = json.loads(data_file.read())

with open(TEST, encoding='utf-8') as data_file2:
    test_data = json.loads(data_file2.read())


def create_training_data():
    print("training data")
    for category in CATEGORIES:
        print(category)
        path = DATADIR  # joins path of directory to each category
        class_num = CATEGORIES.index(category)
        for img in train_data[category]:  # iterate over each image in each category
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])


def create_testing_data():
    print("testing data")
    for category in CATEGORIES:
        print(category)
        path = DATADIR
        class_num = CATEGORIES.index(category)
        for img in test_data[category]:
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            test_data.append([new_array, class_num])


create_training_data()
print(len(training_data))

create_testing_data()
print(len(test_data))

random.shuffle(training_data)
random.shuffle(test_data)

X_train = []
y_train = []

X_test = []
y_test = []

for features, label in training_data:
    X_train.append(features)
    y_train.append(label)

X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

for features, label in test_data:
    X_test.append(features)
    y_test.append(label)

X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("X_train.pickle", "wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle", "wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle", "wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle", "wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
