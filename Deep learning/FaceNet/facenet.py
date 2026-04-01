import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.saving import register_keras_serializable

def create_embedding_network(input_shape=(160,160,3)):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (7,7), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, (5,5), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, (3,3), activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(128)(x)   # final embedding
    return models.Model(inputs, x, name="EmbeddingNet")


@register_keras_serializable(package="Custom", name="triplet_loss")
def triplet_loss(y_true, y_pred, alpha=0.2):
    total_length = y_pred.shape.as_list()[-1]
    anchor = y_pred[:, 0:int(total_length * 1 / 3)]
    positive = y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
    negative = y_pred[:, int(total_length * 2 / 3):int(total_length * 3 / 3)]

    pos_dist = K.sum(K.square(anchor - positive), axis=1)
    neg_dist = K.sum(K.square(anchor - negative), axis=1)
    basic_loss = pos_dist - neg_dist + alpha
    loss = K.maximum(basic_loss, 0.0)
    return loss
