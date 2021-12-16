import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()

print(faces.DESCR)


# viz data
fig, ax = plt.subplots(6, 10, figsize = (10,10), subplot_kw=dict(xticks = [], yticks = []))

for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap = "gray")


X = faces.data
y= faces.target

X.shape


# reduce the 4096 pixel to 100 using pca
from sklearn.decomposition import PCA

model = PCA(100).fit(X)


# viz how well pca with components explains most of X variance
plt.figure(figsize = (10, 10))
plt.plot(np.cumsum(model.explained_variance_ratio_))

plt.xlabel("n components")
plt.ylabel("cummulative variance");


#  helper to plot images used in apply manifold learning helper fn
def plot_components(data, X_new, images = None, ax = None,  thumb_frac = 0.05, cmap = "gray"):
    
    ax = ax or plt.gca()
    ax.plot(X_new[:, 0], X_new[:, 1], ".k")    # plots the manifolds X and Y
    
    if images is not None:
        
        # logic to check how close an image is to an already displayed image on that graph
        # if very close don't display
        min_dist = (thumb_frac * max(X_new.max(0) - X_new.min(0) )) ** 2
        shown_images = np.array( [2 * X_new.max(0)] )
        
        # iters tru images to be displayed and draws an imagebox and draws image within the box
        for i in range(data.shape[0]):  
            dist = np.sum((X_new[i] - shown_images) ** 2, 1)
            
            if np.min(dist) < min_dist:
                continue      # don't show points that are too close together
                
            shown_images = np.vstack([shown_images, X_new[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(
                images[i], cmap = cmap), X_new[i])
            
            ax.add_artist(imagebox)


def apply_manifold_learning(X, method, show_scatter_plot = False):
    
    X_new = method.fit_transform(X) # transforms X to lower dim using the manifold model's fit transform
    
    print("New shape of X: ", X_new.shape)
    print()
    print("Sample X: \n")
    print(pd.DataFrame(X_new).sample(10))
    print()
    
    if show_scatter_plot: 
        # plots scatter plot of images datapoints
        fig,ax = plt.subplots(figsize = (10, 10))
        ax.scatter(X_new[:, 0], X_new[:, 1], cmap = "Spectral")
        plt.xlabel("X[0] after transform")
        plt.ylabel("X[1] after transform")
        
    fig, ax = plt.subplots(figsize = (10, 10))
    plot_components(X[:20, :], X_new[:20, :], images = faces.images[:, ::2, ::2])  # plots actual faces
    plt.xlabel("Component 1")
    plt.ylabel("Component 2");
    
    return method


from sklearn.manifold import MDS

mds = apply_manifold_learning(X, MDS(n_components = 2, metric = True, n_init = 1, max_iter = 100),
                             show_scatter_plot=True)


from sklearn.manifold import Isomap

isomap = apply_manifold_learning(X, Isomap(n_neighbors= 10, n_components = 2))


from sklearn.manifold import LocallyLinearEmbedding

lle = apply_manifold_learning(X, LocallyLinearEmbedding(n_neighbors = 10, n_components = 2, method = "modified"))



















































































































































