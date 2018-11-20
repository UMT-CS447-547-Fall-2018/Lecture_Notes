import keras
from keras.datasets import mnist
import keras.models as km
import keras.layers as kl
from keras import backend as K

batch_size = 256
N = 10
epochs = 24

rows,cols = 28,28
input_shape = (rows,cols,1)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

m_train = X_train.shape[0]
m_test = X_test.shape[0]

X_train = X_train/255
X_test = X_test/255

X_train = X_train.reshape((m_train,rows,cols,1))
X_test = X_test.reshape((m_test,rows,cols,1))

y_train = keras.utils.to_categorical(y_train, N)
y_test = keras.utils.to_categorical(y_test, N)

model = km.Sequential()
model.add(kl.Conv2D(32,kernel_size=(5,5),
                       activation='relu',
                       input_shape=input_shape))
model.add(kl.Conv2D(64,(3,3),activation='relu'))
model.add(kl.MaxPooling2D(pool_size=(2,2)))
model.add(kl.Dropout(0.2))
model.add(kl.Flatten())
model.add(kl.Dense(128,activation='relu'))
model.add(kl.Dropout(0.4))
model.add(kl.Dense(N,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
