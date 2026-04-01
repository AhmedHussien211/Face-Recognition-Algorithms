import os
import cv2
import numpy as np

def load_images_from_directory(base_path, img_size=(100,100)):
    data = {}
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        if os.path.isdir(person_path):
            images = []
            for file in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, file), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                images.append(img)
            data[person] = np.array(images)
    return data

def create_pairs(data):
    persons = list(data.keys())
    pairs, labels = [], []

    # Positive pairs
    for person in persons:
        imgs = data[person]
        for i in range(len(imgs)-1):
            pairs.append([imgs[i], imgs[i+1]])
            labels.append(1)

    # Negative pairs
    for i in range(len(persons)-1):
        p1, p2 = persons[i], persons[i+1]
        pairs.append([data[p1][0], data[p2][0]])
        labels.append(0)

    pairs = np.array(pairs)
    labels = np.array(labels)
    return pairs, labels

