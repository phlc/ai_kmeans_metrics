from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# Função Main
def main():
    #importar base de dados
    iris = datasets.load_iris()
    X = iris.data #instâncias [double, double, double, double]
    y = iris.target #classes corretas [0-2]
    labels = iris.target_names #nomes das classes ['setosa' 'versicolor' 'virginica']
    attributes = iris.feature_names #nome dos atributos ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    
    # Obter valores minimos e maximos
    min_max = [X.min(0), X.max(0)]

    tabela(attributes, min_max)

    # Gerar Agrupador
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    # Obter clusters a que cada instância pertence
    groups = kmeans.predict(X)

    # Obter centroides
    centroids = kmeans.cluster_centers_

    # Criar Grafico sepal length X sepal width
    for i in range(len(X)):
        plt.plot(X[i][0], X[i][1], cor_forma(y[i], groups[i]))

    plt.show()


# Função para criar tabela
def tabela(attributes, data):
    row_headers = ["Mínimo", "Máximo"]
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 1))
    ccolors = plt.cm.BuPu(np.full(len(attributes), 1))
    
    the_table = plt.table(cellText=data,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=attributes,
                        loc='center')  

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.show() 


#função para gerar marcador de forma e cor da instância no gráfico
def cor_forma(y, group):
    # marcador de cor e forma
    marker = ''

    # estabelecer cor em razão do cluster
    if(group == 0):
        marker+='r'
    elif(group ==1):
        marker+='b'
    elif(group==2):
        marker+='y'

    # estabelecer forma em razão da classe
    if(y == 0):
        marker+='^'
    elif(y ==1):
        marker+='o'
    elif(y==2):
        marker+='s'

    return marker



# Chamar Main

main()







