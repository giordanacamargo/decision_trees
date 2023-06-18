from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
import numpy as np                              # Manipulação de arrays
import matplotlib.pyplot as plt                 # Criação de gráficos
from Setup import save_fig


# Cria e visualiza as fronteiras de decisão de dois classificadores de árvore de decisão treinados com o conjunto de dados "make_moons" do sklearn.datasets.
# make_moons é chamada com os parâmetros n_samples=150 (número de amostras), noise=0.2 (ruído nos dados) e random_state=42 (semente aleatória para reprodutibilidade).
X_moons, y_moons = make_moons(n_samples=150, noise=0.2, random_state=42)

# tree_clf1 é criada sem restrições, apenas com random state
tree_clf1 = DecisionTreeClassifier(random_state=42)
# tree_clf2 é criada com restrições (garantirá que cada nó folha tenha no mínimo 5 amostras associadas a ele.)
tree_clf2 = DecisionTreeClassifier(min_samples_leaf=5, random_state=42)

# tree_clf3 é criada com restrições (garantirá que cada nó folha tenha no mínimo 3 amostras associadas a ele.)
tree_clf3 = DecisionTreeClassifier(min_samples_leaf=3, random_state=42)

tree_clf1.fit(X_moons, y_moons)
tree_clf2.fit(X_moons, y_moons)
tree_clf3.fit(X_moons, y_moons)


def plot_decision_boundary(clf, X, y, axes, cmap):
    x1, x2 = np.meshgrid(np.linspace(axes[0], axes[1], 100),
                         np.linspace(axes[2], axes[3], 100))
    X_new = np.c_[x1.ravel(), x2.ravel()]
    y_pred = clf.predict(X_new).reshape(x1.shape)

    plt.contourf(x1, x2, y_pred, alpha=0.3, cmap=cmap)
    plt.contour(x1, x2, y_pred, cmap="Greys", alpha=0.8)
    colors = {"Wistia": ["#78785c", "#c47b27"], "Pastel1": ["red", "blue"]}
    markers = ("o", "^")
    for idx in (0, 1):
        plt.plot(X[:, 0][y == idx], X[:, 1][y == idx],
                 color=colors[cmap][idx], marker=markers[idx], linestyle="none")
    plt.axis(axes)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation=0)


fig, axes = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)

plt.sca(axes[0])
plot_decision_boundary(tree_clf1, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title("Sem restrições: ")

plt.sca(axes[2])
plot_decision_boundary(tree_clf2, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"min_samples_leaf = {tree_clf2.min_samples_leaf}")

plt.sca(axes[1])
plot_decision_boundary(tree_clf3, X_moons, y_moons,
                       axes=[-1.5, 2.4, -1, 1.5], cmap="Wistia")
plt.title(f"min_samples_leaf = {tree_clf3.min_samples_leaf}")
plt.ylabel("")
save_fig("min_samples_leaf_plot")
plt.show()