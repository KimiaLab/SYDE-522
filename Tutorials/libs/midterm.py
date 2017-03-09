import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

def get_data_for_ques1(l=5000, angle=45):

    x = np.random.normal(0, 10, l)
    y = np.random.normal(0, 2, l)

    X = np.array([x, y])

    theta = np.pi * angle / 180

    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])

    ret = np.zeros((len(x), 2))
    d = np.dot(rotation, X)
    ret[:, 0] = d[0] - (-5. + np.mean(d[0]))
    ret[:, 1] = d[1] - (-5. + np.mean(d[1]))

    return ret


def plot_ques1(data, alpha=0.3):
    plt.figure(figsize=(5, 5))

    # plot the data
    plt.scatter(data[:, 0], data[:, 1], s=10, alpha=alpha)
    plt.grid(True)
    plt.xlabel('X1', fontsize=16)
    plt.ylabel('X2', fontsize=16)


def train_pca_on_faces(n_eigen_faces = 50):
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    n_components = n_eigen_faces
    X = lfw_people.data
    return PCA(n_components=n_components, svd_solver='randomized').fit(X)


def plot_eigen_faces(pca):
    plt.figure(figsize=(7, 5))
    for i, component in enumerate(pca.components_):
        plt.subplot(5, 10, i+1)
        plt.imshow(component.reshape(50, 37), cmap='Greys_r')
        plt.axis('off')
