import numpy as np
import matplotlib.pylab as plt
import networkx as nx

def load_2cluster_network():
	""" Generate graph with 12 nodes and 26 edges, with 2 
		main clusters connected by two paths
	"""
	G = nx.Graph()
	for i in range(1, 13):
		G.add_node(i)
	edges_list = [(1, 2), (1, 4), (2, 4), (2, 5), (2, 3),
	(3, 5), (3, 4), (5, 6), (6, 9), (4, 11), (7, 8),
	(7, 9), (8, 10), (8, 9), (7, 10), (9, 10), (10, 11), 
	(11, 12), (10, 12), (6, 2), (6, 3), (6, 1), (9, 11), (9, 12), (4, 6), (11, 7)]
	G.add_edges_from(edges_list)
	print(nx.number_of_nodes(G))
	print(nx.number_of_edges(G))
	return G

def load_3cluster_network():
	""" Generate graph with 12 nodes and 20 edges, with 3 
		main clusters (fully connected subgraphs)
		connected by one path
	"""
	G = nx.Graph()
	for i in range(1, 13):
		G.add_node(i)
	edges_list = [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (2, 4), (2, 7), 
	(5, 7), (5, 6), (5, 8), (6, 7), (6, 8), (7, 8), (7, 9), (9, 10), 
	(9, 11), (9, 12), (10, 11), (10, 12), (11, 12)]
	G.add_edges_from(edges_list)
	print(nx.number_of_nodes(G))
	print(nx.number_of_edges(G))
	return G


def load_cycle_graph(size=5):
	""" Generating a cycle (ring) graph.
		size = number of nodes
	"""
	G = nx.cycle_graph(size)
	return G

def load_arenas_graph():
	""" Generate graph from fig.3 from the paper of Alex Arenas:
		doi:10.1016/j.physd.2006.09.029
	"""
	A = np.array([
		[0, 1, 1, 1, 1, 1, 1, 1, 1],
		[1, 0, 0, 0, 0, 1, 1, 0, 0], 
		[1, 0, 0, 0, 0, 0, 0, 1, 1], 
		[1, 0, 0, 0, 1, 0, 0, 0, 0], 
		[1, 0, 0 ,1, 0, 0, 0, 0, 0], 
		[1, 1, 0, 0, 0, 0, 1, 0, 0], 
		[1, 1, 0, 0, 0, 1, 0, 0, 0], 
		[1, 0, 1, 0, 0, 0, 0, 0, 1], 
		[1, 0, 1, 0, 0, 0, 0, 1, 0]
		])
	G = nx.from_numpy_matrix(A)
	return G

def load_barbell_graph(size=4):
	""" Generate two complete graphs of the given size, 
		connected by a path
	"""
	G = nx.barbell_graph(size, 0)
	return G