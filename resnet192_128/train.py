from comet_ml import Experiment  # must be imported before keras
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import pickle


# check GPU availability
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


img_dir = 'images_cropped/'
metadata_filepath = 'X_meta.csv'
batch_size = 128


# set up experiment logging
# set COMET_API_KEY in your environment variables
# or pass it as the first value in the Experiment object
experiment = Experiment(
    "CgFCfEAIYJVIxez3BZzCqFeeX",
    workspace="ceceshao1", project_name="aleksey-open-fruits"
)


# get X and y values for flow_from_directory
X_meta = pd.read_csv(metadata_filepath)
X = X_meta[['CroppedImageURL']].values
y = X_meta['LabelName'].values


# define data generators
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1/255,
)
train_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(192, 192),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(192, 192),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# define the model

# init a new model with 192x192 CNN layers
# padding='same' will downsample to 96x96
# # which is the expected input size for pretrain
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(192, 192, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# load the pretrained model
prior = load_model('resnet96_128/model-96.h5')

# add all but the first two layers of VGG16 to the new model
# strip the input layer out, this is now 96x96
# also strip out the first convolutional layer, this took the 96x96 input and convolved it but
# this is now the job of the three new layers.
for layer in prior.layers[1:]:
    model.add(layer)

# the pretrained CNN layers are already marked non-trainable
# mark off the top layers as well
for layer in model.layers[-4:]:
    layer.trainable = False
    
# set layer names (otherwise names may collide)
for i, layer in enumerate(model.layers):
    layer.name = f'layer_{i}'

# compile the model
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# generate class weights
import os
labels_count = dict()
for img_class in [ic for ic in os.listdir('images_cropped/') if ic[0] != '.']:
    labels_count[img_class] = len(os.listdir('images_cropped/' + img_class))
total_count = sum(labels_count.values())
class_weights = {cls: total_count / count for cls, count in enumerate(labels_count.values())}


# fit the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(train_generator.filenames) // batch_size,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(patience=2)
    ]
)


# save model artifacts
model.save('/opt/ml/model/model-96.h5')
with open('/opt/ml/model/model-96-history.pickle', 'wb') as fp:
    pickle.dump(history.history, fp)


experiment.end()
