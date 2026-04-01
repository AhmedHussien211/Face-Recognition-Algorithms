import numpy as np

class Eigenfaces:
    def __init__(self, num_components=50):
        self.num_components = num_components
        self.mean_face = None
        self.eigenfaces = None
        self.projections = None
        self.labels = None

    def fit(self, X, y):
        # X: matrix of flattened images (n_samples x n_features)
        self.mean_face = np.mean(X, axis=0)
        X_centered = X - self.mean_face

        # Compute covariance trick
        cov_matrix = np.dot(X_centered, X_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors by eigenvalues
        idx = np.argsort(-eigvals)
        eigvecs = eigvecs[:, idx]

        # Map back to original space
        eigenfaces = np.dot(X_centered.T, eigvecs)
        eigenfaces = eigenfaces[:, :self.num_components]
        self.eigenfaces = eigenfaces

        # Normalize
        for i in range(self.eigenfaces.shape[1]):
            self.eigenfaces[:, i] /= np.linalg.norm(self.eigenfaces[:, i])

        # Project training images
        self.projections = np.dot(X_centered, self.eigenfaces)
        self.labels = y

    def predict(self, x):
        x_centered = x - self.mean_face
        proj = np.dot(x_centered, self.eigenfaces)
        distances = np.linalg.norm(self.projections - proj, axis=1)
        return self.labels[np.argmin(distances)]