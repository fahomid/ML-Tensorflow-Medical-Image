import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import pydicom
import numpy as np
import PIL

# Generate data from dicom file
dataset = [];
labels = [];
for root, dirs, files in os.walk("training_data/Cancer"):
    for file in files:
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(root, file))
            dataset.append(ds.pixel_array)
            labels.append(1);

for root, dirs, files in os.walk("training_data/Normal"):
    for file in files:
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(root, file))
            dataset.append(ds.pixel_array)
            labels.append(0)


dataset_size = len(dataset)
dataset = np.asarray(dataset)
labels = np.asarray(labels)

# create model
model = Sequential()
model.add(Dense(32, activation='tanh', input_shape=(512, 512)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(dataset, labels, epochs=10, shuffle=True, batch_size=32)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("\n\nModel saved to disk\n\n")
model.summary()