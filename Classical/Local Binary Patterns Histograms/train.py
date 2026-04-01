from lbph import LBPH
from utils import load_images_from_folder
import pickle

X, y = load_images_from_folder("../../Datasets/train")

model = LBPH(radius=1, neighbors=8, grid_x=8, grid_y=8)
model.fit(X, y)

with open("lbph_model.pkl", "wb") as f:
    pickle.dump(model, f)
