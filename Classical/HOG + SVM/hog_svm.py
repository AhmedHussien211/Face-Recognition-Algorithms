import cv2
import numpy as np
from skimage.feature import hog
from sklearn import svm

class HOGSVM:
    def __init__(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.clf = svm.SVC(kernel='linear', probability=True)

    def _extract_hog(self, image):
        return hog(image,
                   orientations=self.orientations,
                   pixels_per_cell=self.pixels_per_cell,
                   cells_per_block=self.cells_per_block,
                   block_norm='L2-Hys')

    def fit(self, X, y):
        features = [self._extract_hog(img.reshape(int(np.sqrt(len(img))), -1)) for img in X]
        self.clf.fit(features, y)

    def predict(self, x):
        feat = self._extract_hog(x.reshape(int(np.sqrt(len(x))), -1))
        return self.clf.predict([feat])[0]