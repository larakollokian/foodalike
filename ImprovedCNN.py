from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
import pickle

num_classes = 24
input_shape = 50

train_data = pickle.load(open("X_train.pickle", "rb"))
train_labels = pickle.load(open("y_train.pickle", "rb"))
train_labels_one_hot = to_categorical(train_labels)
test_data = pickle.load(open("X_test.pickle", "rb"))
test_labels = pickle.load(open("y_test.pickle", "rb"))
test_labels_one_hot = to_categorical(test_labels)


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


model1 = create_model()
batch_size = 24
epochs = 100
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

history = model1.fit(train_data, train_labels_one_hot, batch_size=batch_size, epochs=epochs, verbose=1,
                     validation_data=(test_data, test_labels_one_hot))

model1.evaluate(test_data, test_labels_one_hot)

model1.save('food_prediction.model')
