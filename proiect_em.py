import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment
from sklearn.datasets import load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def cluster_accuracy(y_true, y_pred):
    classes_true = np.unique(y_true)
    classes_pred = np.unique(y_pred)

    cost_matrix = np.zeros((len(classes_true), len(classes_pred)), dtype=int)

    for i, true_class in enumerate(classes_true):
        for j, pred_class in enumerate(classes_pred):
            cost_matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    mapping = {}
    for i, j in zip(row_ind, col_ind):
        mapping[classes_pred[j]] = classes_true[i]

    y_pred_mapped = np.array([mapping[label] for label in y_pred])
    acc = accuracy_score(y_true, y_pred_mapped)

    return acc, y_pred_mapped


def run_em_experiment(dataset_name, X, y, n_components):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    em_model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        n_init=10
    )

    em_model.fit(X_scaled)
    y_pred = em_model.predict(X_scaled)

    silhouette = silhouette_score(X_scaled, y_pred)
    ari = adjusted_rand_score(y, y_pred)
    nmi = normalized_mutual_info_score(y, y_pred)
    acc, y_pred_mapped = cluster_accuracy(y, y_pred)

    print(f"\n{'=' * 50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 50}")
    print(f"Numar esantioane: {X.shape[0]}")
    print(f"Numar feature-uri: {X.shape[1]}")
    print(f"Numar componente EM: {n_components}")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"Cluster Accuracy: {acc:.4f}")

    plot_results(dataset_name, X_scaled, y, y_pred, y_pred_mapped)

    return {
        "dataset": dataset_name,
        "samples": X.shape[0],
        "features": X.shape[1],
        "components": n_components,
        "silhouette": silhouette,
        "ari": ari,
        "nmi": nmi,
        "accuracy": acc,
    }


def plot_results(dataset_name, X_scaled, y_true, y_pred, y_pred_mapped):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true)
    plt.title(f"{dataset_name} - Clase reale")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(*scatter1.legend_elements(), title="Clase")

    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_mapped)
    plt.title(f"{dataset_name} - Clustere EM")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(*scatter2.legend_elements(), title="Clustere")

    plt.tight_layout()
    plt.show()


def print_report_summary(results):
    print("\n" + "=" * 60)
    print("SINTEZA FINALA PENTRU RAPORT")
    print("=" * 60)

    for res in results:
        print(f"\nDataset: {res['dataset']}")
        print(f"- Esantioane: {res['samples']}")
        print(f"- Feature-uri: {res['features']}")
        print(f"- Componente EM: {res['components']}")
        print(f"- Silhouette Score: {res['silhouette']:.4f}")
        print(f"- ARI: {res['ari']:.4f}")
        print(f"- NMI: {res['nmi']:.4f}")
        print(f"- Accuracy: {res['accuracy']:.4f}")


def main():
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target

    wine = load_wine()
    X_wine = wine.data
    y_wine = wine.target

    results = []

    results.append(
        run_em_experiment(
            dataset_name="Iris",
            X=X_iris,
            y=y_iris,
            n_components=3
        )
    )

    results.append(
        run_em_experiment(
            dataset_name="Wine",
            X=X_wine,
            y=y_wine,
            n_components=3
        )
    )

    print_report_summary(results)


if __name__ == "__main__":
    main()
