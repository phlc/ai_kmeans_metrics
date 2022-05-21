from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import metrics
import numpy as np

numero_de_clusters = 3
normalizar = True

# Função Main
def main(k):
    #importar base de dados
    iris = datasets.load_iris()
    X = iris.data #instâncias [double, double, double, double]
    y = iris.target #classes corretas [0-2]
    labels = iris.target_names #nomes das classes ['setosa' 'versicolor' 'virginica']
    attributes = iris.feature_names #nome dos atributos ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    
    # Obter valores minimos e maximos
    min_max = [X.min(0), X.max(0)]


    # Normalizar
    if(normalizar):
        for i in range(len(X)):
            for j in range(4):
                X[i][j] = X[i][j]/min_max[1][j]


    tabela(attributes, min_max)

    # Cria boxplot
    fig, ax = plt.subplots()
    ax.set_title("       ".join(attributes), fontsize=8)
    ax.boxplot(X)

    plt.show()

    # Gerar Agrupador
    kmeans = KMeans(n_clusters=k, random_state=0).fit(X)

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

    # Calcular Metricas
    own_db = davies_bouldin(X, groups, centroids)
    metrics_db = metrics.davies_bouldin_score(X, groups)
    own_sc = silhouette_coefficient(X, groups, centroids)
    metrics_sc = metrics.silhouette_score(X,groups)

    col_headers = ["Davies Bouldin Próprio", "Davies Bouldin Metrics", "Silhouette Próprio", "Silhoutte Metrics"]
    ccolors = plt.cm.BuPu(np.full(len(col_headers), 1))
    
    the_table = plt.table(cellText=[[own_db, metrics_db, own_sc, metrics_sc]],
                        colColours=ccolors,
                        colLabels=col_headers,
                        loc='center')  

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.show() 

# Função para calcular a Distância entre dois pontos no espaço 4D
def distancia(x, y):
    return ((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2 + (x[3]-y[3])**2)**(0.5)


# Função para calcular a distância média intra cluster
def distancia_intra(X, groups, centroids, cluster):
    number_instances = 0
    average = 0.0
    for i in range(len(X)):
        if(groups[i] == cluster):
            average += distancia(X[i], centroids[cluster])
            number_instances+=1
    return average/number_instances

# Função para calcular a metrica de Davies_bouldin
def davies_bouldin(X, groups, centroids):
    db = 0.0
    for i in range(len(centroids)):
        Di = 0.0
        for j in range(len(centroids)):
            if(i!=j):
                Rij = (distancia_intra(X, groups, centroids, i) + distancia_intra(X, groups, centroids, j))
                Rij = Rij / distancia(centroids[i], centroids[j])
                if(Di<Rij):
                    Di = Rij
        db += Di
    return db/len(centroids)


# Função para calcular a metrica Silhouette coefficient
def silhouette_coefficient(X, groups, centroids):
    Sc = 0.0
    for i in range(len(X)):
        ai = 0.0
        bi = 0.0
        intra_cluster = 0
        distance_others = [0.0]*len(centroids)
        number_others = [0]*len(centroids)
        distance_others[groups[i]]=float("inf")
        number_others[groups[i]] = 1

        for j in range(len(X)):
            if(i!=j and groups[i] == groups[j]):
                ai += distancia(X[i], X[j])
                intra_cluster+=1
        ai = ai/intra_cluster

        for j in range(len(X)):
            if(groups[i] != groups[j]):
                distance_others[groups[j]] += distancia(X[i], X[j])
                number_others[groups[j]] += 1


        for j in range(len(distance_others)):
            distance_others[j] /= number_others[j]


        bi = min(distance_others)

        Sc += (bi - ai)/ max(ai, bi)

    return Sc/len(X)

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
    elif(group==1):
        marker+='b'
    elif(group==2):
        marker+='y'
    elif(group==3):
        marker+='g'
    elif(group==4):
        marker+='c'
    elif(group==5):
        marker+='m'

    # estabelecer forma em razão da classe
    if(y == 0):
        marker+='^'
    elif(y ==1):
        marker+='o'
    elif(y==2):
        marker+='s'

    return marker



# Chamar Main
main(numero_de_clusters)






