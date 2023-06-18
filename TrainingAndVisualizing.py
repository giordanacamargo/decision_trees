import sys

assert sys.version_info >= (3, 7)
from packaging import version
import sklearn

assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
import matplotlib.pyplot as plt
from pathlib import Path
from graphviz import Source
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import subprocess

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

IMAGES_PATH = Path() / "images" / "decision_trees"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


iris = load_iris(as_frame=True)
X_iris = iris.data[["petal length (cm)", "petal width (cm)"]].values
y_iris = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X_iris, y_iris)

export_graphviz(
    tree_clf,
    out_file=str(IMAGES_PATH / "iris_tree.dot"),  # path differs in the book
    feature_names=["petal length (cm)", "petal width (cm)"],
    class_names=iris.target_names,
    rounded=True,
    filled=True
)


Source.from_file(IMAGES_PATH / "iris_tree.dot")  # path differs in the book

def converter_dot_para_png(caminho_dot, caminho_png):
    graph = graphviz.Source.from_file(caminho_dot)
    graph.format = 'png'
    graph.render(caminho_png, cleanup=True)

caminho_dot = 'C:\\Users\\giord\\OneDrive\\Documentos\\Projetos\\Faculdade\\InteligenciaArtificial\\decision_Trees\\images\\decision_trees\\iris_tree.dot'
caminho_png = 'C:\\Users\\giord\\OneDrive\\Documentos\\Projetos\\Faculdade\\InteligenciaArtificial\\decision_Trees\\images\\decision_trees\\iris_tree.png'

converter_dot_para_png(caminho_dot, caminho_png)