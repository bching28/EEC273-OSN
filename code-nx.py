import csv
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import networkx as nx
from matplotlib import pyplot as plt
import community

def create_graph(x):
    with open (x,'rt') as dataIn:
        dataIn = csv.reader(dataIn)
        headers = next(dataIn)
        data = [row for row in dataIn]

    uniqueData = list(set([row[0] for row in data]))
    id = list(enumerate(uniqueData))

    keys = {node: i for i, node in enumerate(uniqueData)}

    links = []

    for row in data:
        try:
            links.append({keys[row[0]]:keys[row[1]]})
        except:
            links.append({row[0]:row[1]})



    G = nx.Graph()
    dataNodeId = []

    for row in id:
        dataNodeId.append(row[0])


    G.add_nodes_from(dataNodeId)

    for node in links:
        edges = node.items()
        G.add_edge(*edges[0])


    return G


def community_detection(G):
    partition = community.best_partition(G)
    return partition

def betweeness_centrality(G):
    centrality = nx.betweenness_centrality(G)
    return centrality

def drawing(G, partition):
    
    size = float(len(set(partition.values())))
    pos = nx.spring_layout(G)



    count = 0.
    for com in set(partition.values()) :
        count = count + 1.
        list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
        nx.draw_networkx_nodes(G, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))

    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.show()
    return

def main():
    x = '/Users/ishitasharma/Desktop/gephi_output.csv'
    
    Graph = create_graph(x)
    comm = community_detection(Graph)
    cent = betweeness_centrality(Graph)
    drawing(Graph, comm)
    drawing(Graph, cent)


if __name__ == "__main__":
    main()

