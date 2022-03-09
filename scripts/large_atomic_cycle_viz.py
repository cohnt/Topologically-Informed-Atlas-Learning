#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from collections import deque

from src.util import adj_mat_to_adj_list, edge_set_to_adj_mat, maximize_plt_fig
import src.visualization as visualization

seed = 43
np.random.seed(seed)

def find_large_atomic_cycle_with_viz(adj_mat, L):
	# Implementation of the above routine from the CycleCut paper, for
	# graphs stored as adjacency matrices.
	# adj_mat should be a graph, stored as an adjacency matrix. Both distance
	# and connectivity matrices are acceptable. Vertices should not connect
	# to themselves. L is the cycle length hyperparamter
	# Returns a large atomic cycle if one is found, otherwise an empty deque
	#
	# Convert to an adjacency list for faster lookup
	G = adj_mat_to_adj_list(adj_mat)
	#
	# Note: vertices are simply stored as their indices
	# Note: edges are stored as frozen sets of two vertices so a->b = b->a

	# Import visualization variables
	prev_outer_edges = None
	outer_edges = None
	inner_edges_list = []

	Q = deque()
	S = set()
	T = set()
	P = np.zeros(len(G), dtype=int)-1 # List of parent vertices
	v0 = np.random.randint(len(G))
	Q.append(v0)
	S.add(v0)
	outer_edges = T.copy()
	while len(Q) > 0:
		a = Q.popleft()
		for b in G[a]:
			if b in S:
				P[b] = -1 # -1 Represents null
				I = deque()
				U = set()
				I.append(b)
				U.add(b)
				inner_edges_list = []
				inner_edges_list.append(U.copy())
				while len(I) > 0:
					break_out = False
					c = I.popleft()
					for d in G[c]:
						if (d not in U) and ({c,d} in T):
							P[d] = c
							I.append(d)
							U.add(d)
							inner_edges_list.append(U.copy())
							if d == a:
								Y = deque()
								while d != -1:
									Y.append(d)
									d = P[d]
								if len(Y) >= L:
									T.add(frozenset((a,b)))
									prev_outer_edges = outer_edges.copy()
									outer_edges = T.copy()
									return Y, prev_outer_edges, outer_edges, inner_edges_list
								else:
									break_out = True
								break # Break out of for d in G[c]
					if break_out:
						break # Break out of while len(I) > 0
			else:
				Q.append(b)
				S.add(b)
			T.add(frozenset((a,b)))
			prev_outer_edges = outer_edges.copy()
			outer_edges = T.copy()
	return deque(), {}, {}, [{}]

def draw_and_pause():
	plt.draw()
	plt.pause(0.001)
	plt.waitforbuttonpress()

data = np.random.random((500,2))
const = 0.33
data = data[np.logical_or.reduce((data[:,0] < const, data[:,0] > (1-const), np.logical_or(data[:,1] < const, data[:,1] > (1-const))))]
adj_mat = kneighbors_graph(data, 12).toarray()
adj_mat = np.maximum(adj_mat, adj_mat.T)
cycle, prev_outer_edges, outer_edges, inner_edges_list = find_large_atomic_cycle_with_viz(adj_mat, 10)

fig, ax = plt.subplots()
maximize_plt_fig(fig)

# First, draw the data and neighbors
visualization.draw_neighbors(ax, data, adj_mat, c="grey", linewidth=0.5, zorder=1)
ax.scatter(data[:,0], data[:,1], c="grey", zorder=10)
ax.set_title("Data and k-Nearest-Neighbors")
draw_and_pause()

# Now, show the outer BFS right before the cycle was found
prev_outer_edges_adj_mat = edge_set_to_adj_mat(prev_outer_edges, data.shape[0])
visualization.draw_neighbors(ax, data, prev_outer_edges_adj_mat, c="blue", linewidth=2.0, zorder=50)
ax.set_title("Outer BFS (Right Before Finding a Large Atomic Cycle)")
draw_and_pause()

# Now, show the outer BFS when it finds the cycle
new_edges = outer_edges - prev_outer_edges
new_edges_adj_mat = edge_set_to_adj_mat(new_edges, data.shape[0])
visualization.draw_neighbors(ax, data, new_edges_adj_mat, c="green", linewidth=4.0, zorder=50)
ax.set_title("Identified a Cycle")
draw_and_pause()

# Now, visualize each iteration of the inner BFS
point_list = np.array(list(inner_edges_list[0]))
ax.scatter(data[point_list,0], data[point_list,1], c="red", zorder=100)
ax.set_title("Inner BFS (Checking if it's a Large Atomic Cycle)")
plt.draw()
plt.pause(0.001)
every = 10
for i in range(1,len(inner_edges_list), every):
	point_set = set()
	for j in range(i, min(i+every, len(inner_edges_list))):
		point_set = point_set | inner_edges_list[j]
	point_list = np.array(list(point_set - inner_edges_list[i-1]))
	print(len(point_list))
	ax.scatter(data[point_list,0], data[point_list,1], c="red", zorder=100)
	plt.draw()
	plt.pause(0.001)
draw_and_pause()

# Finally, trace the atomic cycle
if len(cycle) > 0:
	visualization.draw_cycle(ax, data, cycle, c="red", zorder=200, linewidth=4.0)
	ax.set_title("Large Atomic Cycle Found")
	draw_and_pause()