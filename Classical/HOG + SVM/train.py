from hog_svm import HOGSVM
from utils import load_images_from_folder
import pickle

X, y = load_images_from_folder("../../Datasets/train")

model = HOGSVM()
model.fit(X, y)

with open("hog_svm_model.pkl", "wb") as f:
    pickle.dump(model, f)
