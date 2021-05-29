import tensorflow as tf
import wget
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

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

#link = "https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
#wget.download(link)
data = "./"
train_link = data + "train.p"
valid_link = data + "valid.p"
test_link = data + "test.p"

# Read datas from files
with open(train_link, mode="rb") as f:
    train = pickle.load(f)

with open(valid_link, mode="rb") as f:
    valid = pickle.load(f)

with open(test_link, mode="rb") as f:
    test = pickle.load(f)

train_X = train["features"] # 34799 * 32 * 32 * 3
train_Y = train["labels"]

#print(train_X[0])
#plt.imshow(train_X[0])
#print(classNames[train_Y[0]])

# Shuffling the datas
train_X, train_Y = shuffle(train_X, train_Y)

valid_X = valid["features"]
valid_Y = valid["labels"]
test_X = test["features"]
test_Y = test["labels"]

# Standardized to decrease impact of data points
train_X = train_X.astype("float") / 255.0
valid_X = valid_X.astype("float") / 255.0
test_X = test_X.astype("float") / 255.0

# Change labels to the form of one-hot encoding
lb = LabelBinarizer()
train_Y = lb.fit_transform(train_Y)
valid_Y = lb.fit_transform(valid_Y)
test_Y = lb.fit_transform(test_Y)
#print(valid_Y[0])

# Build network (follow VGG 16)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD

model = Sequential()

width = 32
height = 32
classes = 43

shape = (width, height, 3)

model.add(Conv2D(32, (3, 3), padding="same", input_shape=shape))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
# We need activation function after use Dense function
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dense(classes))
model.add(Activation("softmax"))

#model.summary()

# Enhance reading images ability
aug = ImageDataGenerator(rotation_range=0.18, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                         horizontal_flip=True)

learning_rate = 0.01

epochs = 10
# epochs
# Steps
#
batch_size = 64

opt = SGD(learning_rate=learning_rate, momentum=0.9)

model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

print("Start Training")
H = model.fit_generator(aug.flow(train_X, train_Y, batch_size=batch_size), validation_data=(valid_X, valid_Y)
                        , steps_per_epoch=train_X.shape[0]/batch_size, epochs=epochs, verbose=1)

#save model
model.save("td_trafficsigns.h5")










