import cv2
import numpy as np

class LBPH:
    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8):
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.histograms = []
        self.labels = []

    def _lbp(self, image):
        lbp = np.zeros_like(image)
        for i in range(self.radius, image.shape[0]-self.radius):
            for j in range(self.radius, image.shape[1]-self.radius):
                center = image[i, j]
                binary = []
                for dy in range(-self.radius, self.radius+1):
                    for dx in range(-self.radius, self.radius+1):
                        if dy == 0 and dx == 0:
                            continue
                        binary.append(1 if image[i+dy, j+dx] >= center else 0)
                lbp[i, j] = int("".join(map(str, binary)), 2)
        return lbp

    def _histogram(self, lbp):
        h, w = lbp.shape
        hist = []
        gx, gy = h // self.grid_x, w // self.grid_y
        for i in range(self.grid_x):
            for j in range(self.grid_y):
                cell = lbp[i*gx:(i+1)*gx, j*gy:(j+1)*gy]
                hist_cell, _ = np.histogram(cell.ravel(), bins=256, range=(0, 256))
                hist.extend(hist_cell)
        return np.array(hist, dtype=np.float32)

    def fit(self, X, y):
        self.histograms = []
        self.labels = y
        for img in X:
            lbp = self._lbp(img.reshape(int(np.sqrt(len(img))), -1))
            hist = self._histogram(lbp)
            self.histograms.append(hist)
        self.histograms = np.array(self.histograms)

    def predict(self, x):
        lbp = self._lbp(x.reshape(int(np.sqrt(len(x))), -1))
        hist = self._histogram(lbp)
        distances = np.linalg.norm(self.histograms - hist, axis=1)
        return self.labels[np.argmin(distances)]