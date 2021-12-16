import pandas as pd
import seaborn as sns
from sklearn import datasets
import matplotlib.pyplot as plt

# to work with data as 3d
from mpl_toolkits import mplot3d


# helper fn to gen s_curve dataset or manifold
def generate_s_curve(method, n = 1500):
    X, color = method(n_samples = n)
    
    print("Shape of X: ", X.shape)
    print("Shape of color: ", color.shape)
    print()
    print("Sample X: \n")
    
    X = pd.DataFrame(X)
    print(X.sample(10))
    
    ax = plt.subplots(figsize = (10, 10))
    ax = plt.axes(projection = "3d")
    
    ax.scatter3D(X[0], X[1], X[2], c = color, cmap = plt.cm.RdYlBu, s= 100)  # plots the 3 X values
    
    return X, color


# gen the complex data point curve
X, color = generate_s_curve(datasets.make_s_curve, 1500)


# helper fn that applies the manifold learning technique and plot result
def apply_manifold_learning(X, method):
    X = method.fit_transform(X)   # method passed is manifold learning tech
    
    print("New shape of X: ", X.shape)
    print()
    print("Sample X: \n")
    
    X= pd.DataFrame(X)
    print(X.sample(10))
    
    plt.subplots(figsize = (10, 10))
    plt.axis("equal")
    
    plt.scatter(X[0], X[1], c = color, cmap = plt.cm.RdYlBu)
    plt.xlabel("X[0]")
    plt.ylabel("X[1]")    
    
    return method


# MDS(Multi Dim Scaling) and n.b: it takes time
from sklearn.manifold import MDS

# red dim of X, while preserving dist btw X instancesx
# when metric = False(non parametric), when in orig data, dist not that important only ranking btw cat is
mds = apply_manifold_learning(X, MDS(n_components= 2, metric = True))


# spectral Embedding
from sklearn.manifold import SpectralEmbedding

spectral_em = apply_manifold_learning(X, 
                                     SpectralEmbedding(n_components = 2, random_state = 0, eigen_solver="arpack"))


#t-SNE(keeps similar dpoints close and dissimilar dpoints apart)

from sklearn.manifold import TSNE

tsne = apply_manifold_learning(X, TSNE(n_components = 2, init="pca", random_state = 0))


#isomap
from sklearn.manifold import Isomap

isomap = apply_manifold_learning(X, Isomap(n_neighbors= 15, n_components = 2))


# LLE(Local linear linear embedding): finds how each d point relates to it's closest neighbours in higher dim
# and tries to preserve this in lower dim space.

from sklearn.manifold import LocallyLinearEmbedding


# method = "standard" doesn't work well on large dataset
lle = apply_manifold_learning(X,
                             LocallyLinearEmbedding(n_neighbors=15, n_components = 2, method = "standard"))


from sklearn.manifold import LocallyLinearEmbedding


# hessian does better than standard but is computationally expensive, best to use "modified"
lle = apply_manifold_learning(X,
                             LocallyLinearEmbedding(n_neighbors=15, n_components = 2, method = "hessian"))

























































































































































