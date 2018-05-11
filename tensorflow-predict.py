from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
import pydicom
import PIL

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

#load data to predict if cancer or not
dataset = []
for root, dirs, files in os.walk("validation_data/test"):
    for file in files:
        if file.endswith(".dcm"):
            ds = pydicom.dcmread(os.path.join(root, file))
            dataset.append(ds.pixel_array)

dataset = np.asarray(dataset)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
score = model.predict(dataset)
for val in score:
    for data in val:
    	print(data)
    	print('\n')
