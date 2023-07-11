import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_biggest_component(G):
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    return G

def singleSourceShortesPaths(G,s):
    sP = {}
    preds = nx.predecessor(G,s)
    for t in G.nodes():
        sP[t] = list(nx.algorithms.shortest_paths.generic._build_paths_from_predecessors([s], t, preds))
    return sP

def allPairsAllShortestPaths(G):
    for s in G.nodes():
        yield s, singleSourceShortesPaths(G, s)

class SearchInformation:
    def __init__(self, G):
        G = get_biggest_component(G)
        G = nx.convert_node_labels_to_integers(G, first_label=0)
        self.G = G

    def compute_probability_shortest_path_matrix(self):
        G = self.G
        N = len(G)
        M_prob_shortest_path = np.zeros((N,N))

        allPairsSP = dict(allPairsAllShortestPaths(G))
        degrees = dict(G.degree)

        for s in allPairsSP.keys():
            for t in allPairsSP[s].keys():
                if s == t:
                    M_prob_shortest_path[s][t] = 1
                    continue

                ks = degrees[s] # grau de s
                kt = degrees[t] # grau de t

                shortestPaths = allPairsSP[s][t]

                TP = 0 # total probabilitie to follow any shortest path

                for path in shortestPaths: # for each shortest path
                    # probabilitie to follow each path
                    P = 1/ks

                    for j in path[:-1]: # for each node 'j' in the way
                    # with j counting all nodes on the path until the last node before the target t is reached. 
                        kj = degrees[j]
                        
                        if kj > 1:
                            P *= 1/(kj - 1)

                    TP += P

                M_prob_shortest_path[s][t] = TP

        return M_prob_shortest_path

    def compute_search_information_matrix(self, probability_shortest_path_matrix):
        #Search information for each pair of nodes
        S = (-1)* np.log2(probability_shortest_path_matrix)
        return S

    def compute_average_search_information(self):
        probability_shortest_path_matrix = self.compute_probability_shortest_path_matrix()
        self.search_information_matrix = self.compute_search_information_matrix(probability_shortest_path_matrix)
        self.average_search_information = np.mean(self.search_information_matrix)

    def get_average_search_information(self):
        return self.average_search_information
