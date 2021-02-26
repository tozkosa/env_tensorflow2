from __future__ import print_function
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from make_tensorboard import make_tensorboard

np.random.seed(1671)

# parameters
NB_EPOCH = 250
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = Adam(lr=0.001)
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
# dropout
DROPOUT = 0.3

(X_train, y_train), (X_test, y_test) = mnist.load_data()

RESHAPED = 784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = to_categorical(y_train, NB_CLASSES)
Y_test = to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, kernel_regularizer=keras.regularizers.l2(0.001),
                input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN, kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_CLASSES, kernel_regularizer=keras.regularizers.l2(0.001)))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

callbacks = [make_tensorboard(set_dir_name='keras_MNIST_V3')]

model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE, epochs=NB_EPOCH, callbacks=callbacks,
          verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test, Y_test, verbose=VERBOSE)
print("\nTest score: ", score[0])
print("Test accuracy: ", score[1])

print("Hello, docker!")