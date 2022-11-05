# https://www.youtube.com/watch?v=YHz0PHcuJnk

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sys import exit

n_samples = 1500
random_state = 170
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
# plt.show()

# Anisotropicly distributed data
transformation = [[.60834549, -.63667341], [-.40887718, .85253229]]
X_aniso = np.dot(X, transformation)

# KMeans with no spectral clustering
y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X_aniso)

# plt.figure(figsize=(4,8))
# plt.subplot(121)
# plt.scatter(X_aniso[:, 0], X_aniso[:, 1], s=20)
# plt.title('Unlabeled data')
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
L_sym = np.identity(len(D)) - M

from scipy.linalg import svd, eig
from sklearn.preprocessing import normalize

# SVD of M : M = U Sigma U^T.
# Usually the decomposition is of the form M = U Sigma V^T but when
# M is normal, then V=U. 
U, Sigma, _ = svd(M, full_matrices=False, lapack_driver='gesvd')
Usubset = U[:, 0:3]
y_pred_sc = KMeans(n_clusters=3).fit_predict(normalize(Usubset))

# Wikipedia algorithim (does not seem to work)
# 1) calc the normalized laplacian
# 2) calc the first k eigenvectors corresponding the k smallest eigenvalues
# 3) consider the matrix formed by the first k eigenvectorsa
# 4) cluster the graph using this matrix by KMeans
w, v = eig(L_sym)
v =  v[:, w.argsort()]
w =  w[w.argsort()]
U_Lsym = v[:, 0:3].real

normalization_constants = np.sqrt(np.sum(np.square(U_Lsym), axis=1)).reshape((-1,1))
U_Lsym = np.multiply(normalization_constants, U_Lsym)


y_pred_sc_eigen = KMeans(n_clusters=3).fit_predict(normalize(U_Lsym))

plt.figure(figsize=(8,4))
plt.subplot(131)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], s=20)

plt.subplot(132)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_sc, s=20)
plt.title('spectral cluster with SVD')

plt.subplot(133)
plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred_sc_eigen, s=20)
plt.title('spectral cluster with Lsym eigen vec\nNg, Jordan and Weiss (2002)')

plt.show()
