from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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

    # Cria boxplot
    fig, ax = plt.subplots()
    ax.set_title("       ".join(attributes), fontsize=8)
    ax.boxplot(X)

    plt.show()

    # Gerar Agrupador
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)

    # Obter clusters a que cada instância pertence
    groups = kmeans.predict(X)

    # Obter centroides
    centroids = kmeans.cluster_centers_


    #Graficos dos Agrupamentos
    
    #Legenda Elementos da Legenda
    legend_elements = [ Line2D([0], [0], marker='^', color='w', label='setosa', markerfacecolor='k', markersize=8),
                        Line2D([0], [0], marker='o', color='w', label='versicolor', markerfacecolor='k', markersize=8),
                        Line2D([0], [0], marker='s', color='w', label='virginica', markerfacecolor='k', markersize=8)]

    
    # Criar Graficos
    for i in range(4):
        for j in range(i+1, 4):
            fig, ax = plt.subplots()
            ax.legend(handles=legend_elements)
            ax.set_xlabel(attributes[i])
            ax.set_ylabel(attributes[j])
            for instancia in range(len(X)):
                ax.plot(X[instancia][i], X[instancia][j], cor_forma(y[instancia], groups[instancia]))

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







