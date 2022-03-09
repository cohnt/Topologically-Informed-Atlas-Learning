import numpy as np
from collections import deque
from src.util import adj_mat_to_adj_list

def find_large_atomic_cycle(adj_mat, L):
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
	Q = deque()
	S = set()
	T = set()
	P = np.zeros(len(G), dtype=int)-1 # List of parent vertices
	v0 = np.random.randint(len(G))
	Q.append(v0)
	S.add(v0)
	while len(Q) > 0:
		a = Q.popleft()
		for b in G[a]:
			if b in S:
				P[b] = -1 # -1 Represents null
				I = deque()
				U = set()
				I.append(b)
				U.add(b)
				while len(I) > 0:
					break_out = False
					c = I.popleft()
					for d in G[c]:
						if (d not in U) and ({c,d} in T):
							P[d] = c
							I.append(d)
							U.add(d)
							if d == a:
								Y = deque()
								while d != -1:
									Y.append(d)
									d = P[d]
								if len(Y) >= L:
									return Y
								else:
									break_out = True
								break # Break out of for d in G[c]
					if break_out:
						break # Break out of while len(I) > 0
			else:
				Q.append(b)
				S.add(b)
			T.add(frozenset((a,b)))
	return deque()