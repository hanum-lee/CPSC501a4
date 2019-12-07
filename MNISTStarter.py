#For Google Collab
#try:
#  # %tensorflow_version only exists in Colab.
#  %tensorflow_version 2.x
#except Exception:
#  pass

import tensorflow as tf

print("--Get data--")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = x_train.reshape(x_train.shape[0],28,28,1)
#x_test = x_test.reshape(x_test.shape[0],28,28,1)
print("--Process data--")
x_train, x_test = x_train / 255.0, x_test / 255.0

input_shape = (28,28)
print("--Make model--")
#Source: https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(28,kernel_size=3,padding='same',activation='relu',input_shape = input_shape),
  tf.keras.layers.MaxPooling1D(pool_size=2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')

  #tf.keras.layers.Flatten(input_shape=(28, 28))
  #tf.keras.layers.Dense(10, activation='softmax'),
  
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(x_train, y_train, epochs=8, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

#Save Model
model.save("MNIST.h5")
