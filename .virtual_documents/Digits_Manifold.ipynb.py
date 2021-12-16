import numpy as np
import pandas as pd
import pylab
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.datasets import load_digits


digits = load_digits()
print(digits.DESCR)


# viz some of the digits
fig, ax = plt.subplots(4, 10, figsize = (8, 8), subplot_kw=dict(xticks = [], yticks=[]))

for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap = "gray")


X = digits.data
y = digits.target

X.shape


# use pc to shrink dim of X from 64 to 50
from sklearn.decomposition import PCA

model = PCA(50).fit(X)


# viz result, via line here line is close to one so will still capture 
# most of the dataset variance
plt.figure(figsize = (8, 8))
plt.plot(np.cumsum(model.explained_variance_ratio_))

pylab.ylim([0, 1.4])

plt.xlabel("n_components")
plt.ylabel("cummulative variance")


# helper function
classes = list(range(10))
target_names = digits.target_names


def apply_manifold_learning(X, y, method):
    
    X = method.fit_transform(X) # transforms X to lower dim using the manifold model's fit transform
    
    print("New shape of X: ", X.shape)
    print()
    print("Sample X: \n")
    print(pd.DataFrame(X).sample(10))
    print()
    
    fig, ax = plt.subplots(figsize = (8, 8))
    # viz transformed data by plot cord of first two dim
    for i, target_name in zip(classes, target_names):
        plt.scatter(X[y == i, 0], X[y == i, 1],
                   label = target_name, cmap = plt.cm.Spectral, s = 100)
        
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.show()
    
    return method


from sklearn.manifold import MDS

msd = apply_manifold_learning(X, y, MDS(n_components = 2, metric = True, n_init = 1, max_iter = 100))


from sklearn.manifold import Isomap

isomap = apply_manifold_learning(X, y, Isomap(n_neighbors=30, n_components = 2))


from sklearn.manifold import LocallyLinearEmbedding

lle = apply_manifold_learning(X, y, LocallyLinearEmbedding(n_neighbors = 30, n_components=2, method ="modified"))


from sklearn.manifold import SpectralEmbedding

spectral_em = apply_manifold_learning(X, y, \
                                     SpectralEmbedding(n_components=2, random_state =0, eigen_solver = "arpack"))


from sklearn.manifold import TSNE

tsne = apply_manifold_learning(X, y, TSNE(n_components=2, init = "pca", random_state = 0))























































































