from utils import  load_images ,create_triplets
from facenet import create_embedding_network,triplet_loss
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

data = load_images("../../Datasets/train")
anchors, positives, negatives = create_triplets(data)
input_shape = (160,160,3)
embedding_net = create_embedding_network(input_shape)

anchor_input = tf.keras.Input(shape=input_shape, name="anchor")
positive_input = tf.keras.Input(shape=input_shape, name="positive")
negative_input = tf.keras.Input(shape=input_shape, name="negative")

anchor_embedding = embedding_net(anchor_input)
positive_embedding = embedding_net(positive_input)
negative_embedding = embedding_net(negative_input)

merged_output = layers.Concatenate(axis=1)([anchor_embedding, positive_embedding, negative_embedding])
model = models.Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_output)

model.compile(optimizer='adam', loss=triplet_loss)


history = model.fit([anchors, positives, negatives],
                    np.zeros(len(anchors)),  # dummy labels
                    batch_size=32, epochs=10)

model.save("facenet.h5")
