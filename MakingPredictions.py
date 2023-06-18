import numpy as np                              # Manipulação de arrays
import matplotlib.pyplot as plt                 # Criação de gráficos
from matplotlib.colors import ListedColormap    # Mapa de cores
from Setup import save_fig
from TrainingAndVisualizing import *


# Cria a lista de cores para a visualização
custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
plt.figure(figsize=(8, 4))

# Duas matrizes bidimensionais, lengths e widths, são criadas utilizando a função np.meshgrid.
# Elas representam todas as possíveis combinações de valores entre 0 e 7.2 para o comprimento e entre 0 e 3 para a largura das pétalas.
lengths, widths = np.meshgrid(np.linspace(0, 7.2, 100), np.linspace(0, 3, 100))

# A variável y_pred recebe as previsões da árvore de decisão (tree_clf.predict) para todos os pontos de X_iris_all.
# As previsões são então remodeladas para ter a mesma forma da matriz lengths.
X_iris_all = np.c_[lengths.ravel(), widths.ravel()]
y_pred = tree_clf.predict(X_iris_all).reshape(lengths.shape)

# É criado um gráfico de contorno preenchido utilizando plt.contourf, passando as matrizes lengths, widths e y_pred.
# A transparência é definida como 0.3 e o mapa de cores utilizado é custom_cmap.
plt.contourf(lengths, widths, y_pred, alpha=0.3, cmap=custom_cmap)

# Itera sobre as classes-alvo das flores no conjunto de dados Iris. Para cada classe,
# é plotado um gráfico de dispersão utilizando plt.plot, mostrando apenas os pontos correspondentes a essa classe.
# A cor e o estilo dos pontos são definidos pela variável style.
for idx, (name, style) in enumerate(zip(iris.target_names, ("yo", "bs", "g^"))):
    plt.plot(X_iris[:, 0][y_iris == idx], X_iris[:, 1][y_iris == idx],
             style, label=f"Iris {name}")

# Definições apenas para fazer o gráfico mais legígel.
tree_clf_deeper = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf_deeper.fit(X_iris, y_iris)
th0, th1, th2a, th2b = tree_clf_deeper.tree_.threshold[[0, 2, 3, 6]]
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.plot([th0, th0], [0, 3], "k-", linewidth=2)
plt.plot([th0, 7.2], [th1, th1], "k--", linewidth=2)
plt.plot([th2a, th2a], [0, th1], "k:", linewidth=2)
plt.plot([th2b, th2b], [th1, 3], "k:", linewidth=2)
plt.text(th0 - 0.05, 1.0, "Depth=0", horizontalalignment="right", fontsize=15)
plt.text(3.2, th1 + 0.02, "Depth=1", verticalalignment="bottom", fontsize=13)
plt.text(th2a + 0.05, 0.5, "(Depth=2)", fontsize=11)
plt.axis([0, 7.2, 0, 3])
plt.legend()
save_fig("decision_tree_decision_boundaries_plot")

plt.show()
