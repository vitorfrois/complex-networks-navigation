import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def get_biggest_component(G):
    G = G.to_undirected()
    G.remove_edges_from(nx.selfloop_edges(G))
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    return G

class SearchInformation:
    def __init__(self, G):
        G = get_biggest_component(G)
        G = nx.convert_node_labels_to_integers(G)
        self.G = G

    def compute_probability_shortest_path_matrix(self):
        G = self.G
        N = len(G)
        M_prob_shortest_path = np.zeros((N,N))
        # all shortest paths
        nodes = list(G.nodes)
        TPs = []

        for s in nodes: # start node
            nodes_except_s = nodes.copy()
            nodes_except_s.remove(s)

            for t in nodes_except_s: # end node
                ks = G.degree[s] # grau de s
                kt = G.degree[t] # grau de t

                # optimization: already have 't' to 's' probability_path
                if M_prob_shortest_path[t-1][s-1] > 1e10:
                    M_prob_shortest_path[s-1][t-1] = ks * (1/kt) * M_prob_shortest_path[t-1][s-1]
                    break

                paths_i_j = list(nx.all_shortest_paths(G,s,t))

                TP = 0 # total probabilitie to follow any shortest path

                ks = G.degree[s] # grau de s

                for path in paths_i_j: # for each shortest path
                    # probabilitie to follow each path
                    P = 1/ks

                    for j in path: # for each node 'j' in the way
                        kj = G.degree[j]
                        # if that node has mode than one neighbor
                        if kj > 1:
                            P *= 1/(kj - 1)

                    TP += P

                M_prob_shortest_path[s-1][t-1] = TP

        np.fill_diagonal(M_prob_shortest_path,1) # Probability to go from node to itself is 1 

        return M_prob_shortest_path

    def compute_search_information_matrix(self, probability_shortest_path_matrix):
        #Search information for each pair of nodes
        S = (-1)* np.log2(probability_shortest_path_matrix)
        return S

    def compute_average_search_information(self):
        probability_shortest_path_matrix = self.compute_probability_shortest_path_matrix()
        self.search_information_matrix = self.compute_search_information_matrix(probability_shortest_path_matrix)
        self.average_search_information = np.mean(self.search_information_matrix)

    def average_search_information(self):
        return self.average_search_information
    
G = nx.barabasi_albert_graph(10, 1)
g_info = SearchInformation(G)
g_info.compute_average_search_information()
print(g_info.average_search_information)