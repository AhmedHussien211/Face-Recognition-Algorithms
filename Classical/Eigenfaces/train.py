from eigenfaces import Eigenfaces
from utils import load_images_from_folder

# Load dataset
X, y = load_images_from_folder("../../Datasets/train")

# Train model
model = Eigenfaces(num_components=50)
model.fit(X, y)

# Save model (optional: pickle)
import pickle
with open("eigenfaces_model.pkl", "wb") as f:
    pickle.dump(model, f)
