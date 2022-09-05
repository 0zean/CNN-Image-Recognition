# Convolutional Neural Network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.995:
            print("\nReached 99.5% accuracy: cancelling training!")
            self.model.stop_training = True


# Part 1 - Data Preprocessing

# Preprocessing the Training set by rescaling to normalize images
train_datagen = ImageDataGenerator(rescale=1/255)

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(128, 128), batch_size=32, class_mode='binary')

# Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1/255)

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(128, 128), batch_size=32, class_mode='binary')


# Part 2 - Building the CNN

# Initialising the CNN
cnn = tf.keras.models.Sequential([

    # 1st Convolution + Pooling
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(128, 128, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Adding a 2nd convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Adding a 3rd convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flattening
    tf.keras.layers.Flatten(),

    # Hidden layer
    tf.keras.layers.Dense(128, activation='relu'),

    # Output Layer
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Part 3 - Training the CNN

# Compiling the CNN
cnn.compile(loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(training_set, 
    epochs=25, 
    verbose=1, 
    validation_data=test_set,
    callbacks=[myCallback()])


# Part 4 - Making a single prediction
import numpy as np
from keras.preprocessing import image

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
