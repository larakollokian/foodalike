import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
from keras.utils import to_categorical

X = pickle.load(open("X.pickle", "rb"))
y_int = pickle.load(open("y.pickle", "rb"))

y_binary = to_categorical(y_int)

X = X / 255.0

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(24))
model.add(Activation("sigmoid"))
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y_binary, batch_size=24, epochs=10, validation_split=0.1)

pickle.dump(model, open("food_classifier.pickle", "wb"))

# model.save('food_predict.model')
