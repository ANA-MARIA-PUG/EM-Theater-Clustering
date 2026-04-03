import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris, make_blobs

# 1. Pregătirea seturilor de date
# Iris: set natural (luăm primele 2 coloane pentru vizualizare 2D)
X_iris = load_iris().data[:, :2]
# Blobs: set sintetic generat controlat
X_blobs, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

datasets = [X_iris, X_blobs]
titles = ['Eșantion Iris (Date Naturale)', 'Eșantion Blobs (Date Sintetice)']
components = [3, 4]  # 3 specii pentru Iris, 4 centre pentru Blobs

# Crearea ferestrei cu două diagrame
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

print("--- REZULTATE PENTRU CONSOLĂ ---")

for i, X in enumerate(datasets):
    # Aplicarea algoritmului EM (GMM) [cite: 12, 14, 15]
    gmm = GaussianMixture(n_components=components[i], n_init=10, random_state=42)
    gmm.fit(X)
    labels = gmm.predict(X)

    # Afișarea rezultatelor în consolă pentru Capitolul 5
    print(f"\nRezultate pentru {titles[i]}:")
    print(f"- Algoritmul a convergut: {gmm.converged_}")
    print(f"- Număr de iterații: {gmm.n_iter_}")
    print(f"- Log-verosimilitate finală: {gmm.lower_bound_:.4f}")

    # Generarea diagramei [cite: 16, 18, 19]
    scatter = axes[i].scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', edgecolors='k')
    axes[i].set_title(f"{titles[i]}\nConvergență în {gmm.n_iter_} iterații")

    # Etichetarea axelor (X și Y)
    axes[i].set_xlabel("Componenta / Atribut 1")
    axes[i].set_ylabel("Componenta / Atribut 2")

plt.tight_layout()
plt.show()