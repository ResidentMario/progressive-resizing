from comet_ml import Experiment  # must be imported before keras
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf


# check GPU availability
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


img_dir = 'images_cropped/'
metadata_filepath = 'X_meta.csv'
batch_size = 256


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
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
)
validation_generator = train_datagen.flow_from_directory(
    img_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


# define the model
prior = keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=(48, 48, 3)
)
model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(256, activation='relu', name='Dense_Intermediate'))
model.add(Dropout(0.1, name='Dropout_Regularization'))
model.add(Dense(12, activation='sigmoid', name='Output'))


# freeze the vgg16 model
for cnn_block_layer in model.layers[0].layers:
    cnn_block_layer.trainable = False
model.layers[0].trainable = False

    
# compile the model
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# fit the model
model.fit_generator(
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


# save model artifact
model.save('/opt/ml/model/model-48.h5')


experiment.end()
