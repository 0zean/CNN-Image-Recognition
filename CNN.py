# Convolutional Neural Network

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import warnings
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

warnings.filterwarnings('ignore')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy: cancelling training!")
            self.model.stop_training = True


# Part 1 - Data Preprocessing

# Preprocessing the Training set by rescaling and augmenting the images
train_datagen = ImageDataGenerator(rescale=1./255.,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(150, 150), batch_size=32, class_mode='binary')

# Preprocessing the Test set
validation_datagen = ImageDataGenerator(rescale=1./255.)

validation_set = validation_datagen.flow_from_directory('dataset/test_set', target_size=(150, 150), batch_size=32, class_mode='binary')


# Part 2 - Transfer Learning to increase accuracy

# Instantiate InceptionV3 model using pre-trained weights
local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


def create_pre_trained_model(weights):
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)
    pre_trained_model.load_weights(weights)
    for layer in pre_trained_model.layers:
        layer.trainable = False
    return pre_trained_model


pre_trained_model = create_pre_trained_model(local_weights_file)


def last_layer_output(pre_trained_model):
    last_desired_layer = pre_trained_model.get_layer('mixed7')
    last_output = last_desired_layer.output
    return last_output


last_output = last_layer_output(pre_trained_model)


# Part 3 - create combined model from InceptionV3 into CNN
def combined_model(pre_trained_model, last_output):
    x = layers.Flatten()(last_output)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=pre_trained_model.input, outputs=x)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model


model = combined_model(pre_trained_model, last_output)

history = model.fit(training_set,
                    validation_data=validation_set,
                    epochs=15,
                    verbose=1,
                    callbacks=[myCallback()])


# Part 4 - View model metrics
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.savefig('Training_and_validation_accuracy.png')
plt.show()


# Part 5 - Making a single prediction

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
print(prediction)
