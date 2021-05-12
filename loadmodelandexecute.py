

import numpy as np


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import numpy as np
from keras.preprocessing import image
import pathlib

model = tf.keras.models.load_model('my_model.h5')

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.summary()


class_names=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']


img_width, img_height = 180, 180
img = image.load_img('sf.jpg', target_size = (img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)


predictions = model.predict(img)
score = tf.nn.softmax(predictions[0])



print("This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score)))



