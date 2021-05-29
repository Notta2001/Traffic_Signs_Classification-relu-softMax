import tensorflow as tf
import pickle

classNames = {0: 'Speed limit (20km/h)',
 1: 'Speed limit (30km/h)',
 2: 'Speed limit (50km/h)',
 3: 'Speed limit (60km/h)',
 4: 'Speed limit (70km/h)',
 5: 'Speed limit (80km/h)',
 6: 'End of speed limit (80km/h)',
 7: 'Speed limit (100km/h)',
 8: 'Speed limit (120km/h)',
 9: 'No passing',
 10: 'No passing for vehicles over 3.5 metric tons',
 11: 'Right-of-way at the next intersection',
 12: 'Priority road',
 13: 'Yield',
 14: 'Stop',
 15: 'No vehicles',
 16: 'Vehicles over 3.5 metric tons prohibited',
 17: 'No entry',
 18: 'General caution',
 19: 'Dangerous curve to the left',
 20: 'Dangerous curve to the right',
 21: 'Double curve',
 22: 'Bumpy road',
 23: 'Slippery road',
 24: 'Road narrows on the right',
 25: 'Road work',
 26: 'Traffic signals',
 27: 'Pedestrians',
 28: 'Children crossing',
 29: 'Bicycles crossing',
 30: 'Beware of ice/snow',
 31: 'Wild animals crossing',
 32: 'End of all speed and passing limits',
 33: 'Turn right ahead',
 34: 'Turn left ahead',
 35: 'Ahead only',
 36: 'Go straight or right',
 37: 'Go straight or left',
 38: 'Keep right',
 39: 'Keep left',
 40: 'Roundabout mandatory',
 41: 'End of no passing',
 42: 'End of no passing by vehicles over 3.5 metric tons'}
data = "./"
train_link = data + "train.p"
valid_link = data + "valid.p"
test_link = data + "test.p"

with open(train_link, mode="rb") as f:
    train = pickle.load(f)

with open(valid_link, mode="rb") as f:
    valid = pickle.load(f)

with open(test_link, mode="rb") as f:
    test = pickle.load(f)

train_X = train["features"] # 34799 * 32 * 32 * 3
train_Y = train["labels"]
valid_X = valid["features"]
valid_Y = valid["labels"]
test_X = test["features"]
test_Y = test["labels"]
train_X = train_X.astype("float") / 255.0
valid_X = valid_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0


saved_model = tf.keras.models.load_model("td_trafficsigns.h5")
result = saved_model.predict(test_X[100:101])

import numpy as np
final = np.argmax(result)
print(classNames[final])

import matplotlib.pyplot as plt
plt.imshow(test_X[100])
plt.show()
