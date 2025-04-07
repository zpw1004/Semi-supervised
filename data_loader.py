import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_and_preprocess_data(args):
    data = pd.read_excel(args.dataset)
    data = data[(data != -9999).all(axis=1)]
    X = data.drop(columns=["Facies", "TYPE", "DEPTH", "class"])
    y = data["class"].values
    depths = data["DEPTH"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state)

    return X_train, X_test, y_train, y_test, depths

def weak_augmentation(X, alpha=None, args=None):
    alpha = alpha or args.weak_alpha
    n_samples = X.shape[0]
    augmented_data = X.clone()
    for i in range(n_samples):
        perturbation = alpha * torch.randn_like(X[i])
        augmented_data[i] += perturbation
    return augmented_data

def strong_augmentation(X, alpha=None, args=None):
    alpha = alpha or args.strong_alpha
    n_samples = X.shape[0]
    augmented_data = X.clone()
    for i in range(n_samples):
        perturbation = alpha * torch.randn_like(X[i])
        augmented_data[i] += perturbation
        augmented_data[i] *= (1 + np.random.uniform(args.scale_min, args.scale_max))
    return augmented_data

def compute_feature_and_depth_similarity(X, depths, args, t1=None, t2=None, mu=None):
    t1 = t1 or args.t1
    t2 = t2 or args.t2
    mu = mu or args.mu
    n_samples = X.shape[0]
    W1 = np.zeros((n_samples, n_samples))
    W2 = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.abs(X[i] - X[j])
            W1[i, j] = W1[j, i] = np.exp(-np.sum(dist) / (4 * t1 ** 2))

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = abs(depths[i] - depths[j])
            W2[i, j] = W2[j, i] = np.exp(-dist / (4 * t2))
    W_combined = mu * W1 + (1 - mu) * W2
    return W_combined