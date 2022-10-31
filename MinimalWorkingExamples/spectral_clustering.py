# https://www.youtube.com/watch?v=YHz0PHcuJnk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
# plt.show()

# Anisotropicly distributed data
transformation = [[.60834549, -.63667341], [-.40887718, .85253229]]
X_aniso = np.dot(X, transformation)

# plt.figure(figsize=(4,8))
# plt.subplot(121)
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], s=20)
# plt.title('Unlabeled data')

y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

# plt.subplot(122)
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred, s=20)
# plt.title('Labels returned by KMeans')
# plt.show()

from scipy.spatial import distance

rbf_param = 7.6
K = np.exp(-rbf_param * distance.cdist(X_aniso, X_aniso, metric='sqeuclidean'))

D = K.sum(axis=1)
D = np.sqrt(1/D)
M = np.multiply(D.reshape((-1,1)), np.multiply(K, D.reshape((-1,1))))

from scipy import linalg
from sklearn.preprocessing import normalize

# EVD of M : M = U Sigma U^T.
# Usually the decomposition is of the form M = U Sigma V^T but when
# M is normal, then V=U. 

U, Sigma, _ = linalg.svd(M, full_matrices=False, lapack_driver='gesvd')
Usubset = U[:, 0:3]
y_pred_sc = KMeans(n_clusters=3).fit_predict(normalize(Usubset))

plt.figure(figsize=(8,4))
plt.subplot(121)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], s=20)

plt.subplot(122)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_sc, s=20)
plt.show()
