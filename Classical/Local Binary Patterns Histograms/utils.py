import cv2
import numpy as np
import os

def load_images_from_folder(folder, size=(100, 100)):
    images, labels = [], []
    for label in os.listdir(folder):
        path = os.path.join(folder, label)
        if os.path.isdir(path):
            for file in os.listdir(path):
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, size)
                    images.append(img.flatten())
                    labels.append(label)
    return np.array(images), np.array(labels)