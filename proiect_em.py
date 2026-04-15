import matplotlib.pyplot as plt
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Seturi de date: mic și mare
X_small, _ = make_blobs(n_samples=50, centers=4, cluster_std=0.60, random_state=0)
X_large, _ = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=0)

datasets = [X_small, X_large]
titles = ['Set Mic (50 de puncte)', 'Set Mare (500 de puncte)']
components = [4, 4]  # Pentru ambele, numărul de componente

# Crearea ferestrei cu două diagrame
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

print("--- REZULTATE PENTRU CONSOLĂ ---")

for i, X in enumerate(datasets):
    # Aplicarea algoritmului EM (GMM)
    gmm = GaussianMixture(n_components=components[i], n_init=10, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)

    # Rezultate în consolă
    print(f"\nRezultate pentru {titles[i]}:")
    print(f"- Algoritmul a convergut: {gmm.converged_}")
    print(f"- Număr de iterații: {gmm.n_iter_}")
    print(f"- Log-verosimilitate finală: {gmm.lower_bound_:.4f}")

    # Diagrama
    ax = axes[i]
    colors = plt.get_cmap('tab10', components[i])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap=colors, edgecolor='k', alpha=0.8)
    ax.set_title(f"{titles[i]}\nConvergență în {gmm.n_iter_} iterații", fontsize=14)
    ax.set_xlabel("Componenta / Atribut 1", fontsize=12)
    ax.set_ylabel("Componenta / Atribut 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Adăugare centre cluster
    centers = gmm.means_
    ax.scatter(centers[:,0], centers[:,1], c='red', s=150, marker='X', label='Centrele Clusterelor')
    ax.legend(fontsize=12)

plt.tight_layout()
plt.show()
