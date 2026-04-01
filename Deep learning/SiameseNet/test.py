from utils import load_images_from_directory , create_pairs
from siamesenet import euclidean_distance, contrastive_loss
import tensorflow as tf
import numpy as np


from tensorflow.keras.models import load_model

model = load_model("siamesenet.h5",
                   custom_objects={"euclidean_distance": euclidean_distance,
                                   "contrastive_loss": contrastive_loss})

data = load_images_from_directory("../../Datasets/test")

pairs, labels = create_pairs(data)
X1 = pairs[:,0].reshape(-1,100,100,1)
X2 = pairs[:,1].reshape(-1,100,100,1)



y_pred = model.predict([X1, X2])
threshold = 0.5
y_pred_class = (y_pred < threshold).astype("int")

accuracy = np.mean(y_pred_class.flatten() == labels)
print("Accuracy:", accuracy)