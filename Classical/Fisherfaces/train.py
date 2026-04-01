from fisherface import Fisherfaces
from utils import load_images_from_folder
import pickle

X, y = load_images_from_folder("../../Datasets/train")

model = Fisherfaces(num_components_pca=100)
model.fit(X, y)

with open("fisherfaces_model.pkl", "wb") as f:
    pickle.dump(model, f)
