import os, cv2, numpy as np

def load_images(base_path, img_size=(160,160)):
    data = {}
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        if os.path.isdir(person_path):
            images = []
            for file in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, file))
                img = cv2.resize(img, img_size)
                images.append(img)
            data[person] = np.array(images)
    return data

def create_triplets(data):
    persons = list(data.keys())
    anchors, positives, negatives = [], [], []

    for person in persons:
        imgs = data[person]
        if len(imgs) < 2:  # need at least 2 images for anchor+positive
            continue
        for i in range(len(imgs)-1):
            anchor = imgs[i]
            positive = imgs[i+1]
            # pick a random negative from another person
            neg_person = np.random.choice([p for p in persons if p != person])
            negative = data[neg_person][0]

            anchors.append(anchor)
            positives.append(positive)
            negatives.append(negative)

    return np.array(anchors), np.array(positives), np.array(negatives)



