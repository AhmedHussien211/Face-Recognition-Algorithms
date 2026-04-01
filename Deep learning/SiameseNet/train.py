from utils import load_images_from_directory, create_pairs
import numpy as np
from siamesenet import create_base_network,euclidean_distance,contrastive_loss
import tensorflow as tf
from tensorflow.keras import layers, models

data = load_images_from_directory("../../Datasets/train")

pairs, labels = create_pairs(data)
X1 = pairs[:,0].reshape(-1,100,100,1)
X2 = pairs[:,1].reshape(-1,100,100,1)

# Build Siamese network
input_shape = (100, 100, 1)  # grayscale faces resized
base_network = create_base_network(input_shape)

input_a = tf.keras.Input(shape=input_shape)
input_b = tf.keras.Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])
model = models.Model([input_a, input_b], distance)

model.compile(loss=contrastive_loss, optimizer='adam')

history = model.fit([X1, X2], labels,
                    batch_size=32, epochs=5, validation_split=0.2)

model.save("siamesenet.h5")




