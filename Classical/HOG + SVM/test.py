import pickle
from utils import load_images_from_folder

with open("hog_svm_model.pkl", "rb") as f:
    model = pickle.load(f)

X_test, y_test = load_images_from_folder("../../Datasets/test")

correct = 0
for i in range(len(X_test)):
    pred = model.predict(X_test[i])
    if pred == y_test[i]:
        correct += 1

print(f"Accuracy: {correct/len(X_test)*100:.2f}%")
