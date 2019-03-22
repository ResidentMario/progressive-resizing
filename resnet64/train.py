from comet_ml import Experiment  # must be imported before keras
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# check GPU availability
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


img_dir = 'images_cropped/'
metadata_filepath = 'X_meta.csv'


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
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(64, 64),
    batch_size=128,
    class_mode='categorical',
    subset='validation'
)


# get model prior
# TODO: validate that this will actually work
# Use the "temp-scratchpad.ipyn b" notebook in the SageMaker instance to do so.
from keras.models import load_model
prior = load_model('model-48.h5')


# define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(prior)


# freeze the vgg16 model
for cnn_block_layer in model.layers[2].layers:
    cnn_block_layer.trainable = False


# compile the model
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# fit the model
batch_size = 128
model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=len(train_generator.filenames) // batch_size,
    callbacks=[
        ReduceLROnPlateau(monitor='val_loss', patience=2),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]
)


# save model artifact
model.save('/opt/ml/model/model-64.h5')


experiment.end()
