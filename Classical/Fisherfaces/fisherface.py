import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class Fisherfaces:
    def __init__(self, num_components_pca=100):
        self.num_components_pca = num_components_pca
        self.mean_face = None
        self.pca_components = None
        self.lda = None
        self.projections = None
        self.labels = None

    def fit(self, X, y):
        # Mean face
        self.mean_face = np.mean(X, axis=0)
        X_centered = X - self.mean_face

        # PCA step
        cov_matrix = np.dot(X_centered, X_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(-eigvals)
        eigvecs = eigvecs[:, idx]
        eigenfaces = np.dot(X_centered.T, eigvecs)
        eigenfaces = eigenfaces[:, :self.num_components_pca]

        # Normalize
        for i in range(eigenfaces.shape[1]):
            eigenfaces[:, i] /= np.linalg.norm(eigenfaces[:, i])

        self.pca_components = eigenfaces
        X_pca = np.dot(X_centered, self.pca_components)

        # LDA step
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(X_pca, y)

        # Store projections
        self.projections = self.lda.transform(X_pca)
        self.labels = y

    def predict(self, x):
        x_centered = x - self.mean_face
        x_pca = np.dot(x_centered, self.pca_components)
        proj = self.lda.transform([x_pca])
        distances = np.linalg.norm(self.projections - proj, axis=1)
        return self.labels[np.argmin(distances)]