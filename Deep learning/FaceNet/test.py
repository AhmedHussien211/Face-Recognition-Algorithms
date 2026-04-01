from tensorflow.keras.models import load_model
from utils import load_images,create_triplets
from facenet import triplet_loss,create_embedding_network

model = load_model("facenet.h5",
                   custom_objects={"triplet_loss": triplet_loss})

# Example: dataset/test/personX/img.jpg
test_data = load_images("../../Datasets/test")   # reuse your load_images function
anchors, positives, negatives = create_triplets(test_data)

embedding_net = model.get_layer("EmbeddingNet")  # name from earlier definition

emb_a = embedding_net.predict(anchors)
emb_p = embedding_net.predict(positives)
emb_n = embedding_net.predict(negatives)

import numpy as np

# Positive distances (anchor vs positive)
pos_dist = np.linalg.norm(emb_a - emb_p, axis=1)

# Negative distances (anchor vs negative)
neg_dist = np.linalg.norm(emb_a - emb_n, axis=1)

# Simple accuracy: positives should be closer than negatives
accuracy = np.mean(pos_dist < neg_dist)
print("Test Accuracy:", accuracy)

