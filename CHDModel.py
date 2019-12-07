#source: https://www.tensorflow.org/tutorials/load_data/csv


import tensorflow as tf
import numpy as np
import pandas as pd



rawdata = pd.read_csv("heart.csv")
#print(rawdata)
csvcolumns = ["row.names","sbp","tobacco","ldl","adiposity","famhist","typea","obesity", "alcohol","age","chd"]
selected_column = ["sbp","tobacco","ldl","adiposity","famhist","typea","obesity", "alcohol","age",]

LABEL_COL = ["chd"]
LABELS = [0,1]

def get_dataset(file_path, **kwargs):
  dataset = tf.data.experimental.make_csv_dataset(
      file_path,
      batch_size=5, # Artificially small to make examples easier to show.
      #label_name=LABEL_COL,
      na_value="?",
      num_epochs=1,
      ignore_errors=True, 
      **kwargs)
  return dataset

def show_batch(dataset):
    for batch in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key,value.numpy()))

def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label

#rawtraindataX = get_dataset("heart_train.csv")
#show_batch(rawtraindataX)
train_X = get_dataset("heart_train.csv",select_columns=selected_column)
show_batch(train_X)
train_Y = get_dataset("heart_train.csv",select_columns=LABEL_COL)
show_batch(train_Y)

test_X = get_dataset("heart_test.csv",select_columns=selected_column)
show_batch(test_X)
test_Y = get_dataset("heart_test.csv",select_columns=LABEL_COL)
show_batch(test_Y)

example_batch = next(iter(train_X)) 

packed_dataset = test_X.map(pack)

for features, labels in packed_dataset.take(1):
  print(features.numpy())
  print()
  print(labels.numpy())


'''model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(28,kernel_size=3,padding='same',activation='relu',input_shape = 9),
  tf.keras.layers.MaxPooling1D(pool_size=3),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')

  #tf.keras.layers.Flatten(input_shape=(28, 28))
  #tf.keras.layers.Dense(10, activation='softmax'),
  
])
model.compile(optimizer='adam', loss='binaryCrossEntropy', metrics=['accuracy'])

print("--Fit model--")
model.fit(train_X, train_Y, epochs=8, verbose=2)

print("--Evaluate model--")
model_loss, model_acc = model.evaluate(test_X,  test_Y, verbose=2)
print(f"Model Loss:    {model_loss:.2f}")
print(f"Model Accuray: {model_acc*100:.1f}%")

#Save Model
model.save("MNIST.h5")'''
