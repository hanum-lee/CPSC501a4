#Some of the ideas were taken from the source.
#Source: https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist

import tensorflow as tf
import numpy as np

print("--Get data--")
with np.load("notMNIST.npz", allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

#x_train = x_train.reshape(x_train.shape[0],28,28,1)
#x_test = x_test.reshape(x_test.shape[0],28,28,1)

print("--Process data--")
print(len(y_train))
x_train, x_test = x_train / 255.0, x_test / 255.0
input_shape = (28,28)

print("--Make model--")
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(28,kernel_size=3,padding='same',activation='relu',input_shape = input_shape),
  tf.keras.layers.MaxPooling1D(pool_size=2),
  tf.keras.layers.Conv1D(28,kernel_size=3,padding='same',activation='relu',input_shape = (14,14)),
  tf.keras.layers.MaxPooling1D(pool_size=2),
  tf.keras.layers.Conv1D(28,kernel_size=3,padding='same',activation='relu',input_shape = (7,7)),
  tf.keras.layers.MaxPooling1D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=10, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")
model.save("notMNIST.h5")
