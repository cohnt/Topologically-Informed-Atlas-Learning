import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def adj_mat_to_adj_list(adj_mat):
	return [list(np.where(adj_mat[i] != 0)[0]) for i in range(adj_mat.shape[0])]

def edge_contraction(adj_mat, i, j):
	# Returns the new adjacency matrix, after contracting the edge from i to j.
	# adj_mat should be a connectivity matrix, where adj_mat[i,j] == 1 iff i and j
	# have an edge between them. If adj_mat is (n,n), the output will be (n-1,n-1)
	# This function does NOT check if there's an edge between i and j, so don't
	# call it unless you know adj_mat[i,j] == 1!

	# We do this by contracting j into i. That is, we delete the jth row and column,
	# and then add edges i-k for each previous edge j-k.
	# This is implemented by first asserting that i<j, and swapping them if they aren't.
	# Then we find all edges out of j, delete the i-j edge, and then for j-k edges, if
	# k>j, decrementing k because we're deleting the jth entry.
	if i > j:
		i, j = j, i
	out_mat = np.delete(np.delete(adj_mat, j, axis=0), j, axis=1)
	prev_edges = np.nonzero(adj_mat[j])[0]
	ij_idx = np.searchsorted(prev_edges, i) # Find the index of the ij edge
	prev_edges = np.delete(prev_edges, ij_idx) # Delete the ij edge
	prev_edges[prev_edges > j] = prev_edges[prev_edges > j] - 1
	out_mat[i,prev_edges] = 1
	out_mat[prev_edges,i] = 1
	return out_mat

def adj_mat_to_edge_set(adj_mat):
	# Takes in an adjacency matrix, and outputs a set of edges. (Note that this
	# discards distance information.)
	nz = np.nonzero(np.triu(adj_mat))
	s = set()
	for i in range(len(nz[0])):
		s.add(frozenset((nz[0][i], nz[1][i])))
	return s

def edge_set_to_adj_mat(edge_set, n_points):
	# Takes in a set of edges, and the number of points, and outputs an adjacency
	# matrix in connectivity form.
	adj_mat = np.zeros((n_points, n_points), dtype=int)
	for edge in edge_set:
		adj_mat[tuple(edge)] = 1
	adj_mat = np.maximum(adj_mat, adj_mat.T)
	return adj_mat

def n_walk_closure(adj_mat, n):
	# Given an adjacency matrix (in connectivity form), this outputs the n-walk
	# closure. For example, if n=2, then a-b and b-c will mean a-c is added as
	# an edge. This is inclusive of lower length walks.
	new_mat = adj_mat.copy()
	running_mat = adj_mat.copy()
	for i in range(2, n+1):
		running_mat = np.matmul(running_mat, adj_mat)
		new_mat += running_mat
	return (new_mat > 0).astype(int)

def maximize_plt_fig(fig):
	# Reference:
	# https://stackoverflow.com/questions/12439588/how-to-maximize-a-plt-show-window-using-python/18824814#18824814
	# https://stackoverflow.com/questions/13065753/obtaining-the-figure-manager-via-the-oo-interface-in-matplotlib
	# Works for the Qt backend. Call before plt.show() or plt.draw().
	fig.canvas.manager.window.showMaximized()

def get_subfolder_names(path):
	# Returns a list of subfolder names in the directory specified by path.
	# Source: https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
	return [f.name for f in os.scandir(path) if f.is_dir()]

def quit_on_q(fig):
	# Modifies a matplotlib figure object so that the program halts if you press q.
	fig.canvas.mpl_connect("key_press_event", _quit_on_q_helper)

def _quit_on_q_helper(event):
	if event.key == "q":
		exit(0)

def strong_sharpen_func(img):
	return cv2.addWeighted(img, 4, cv2.blur(img, (25, 25)), -4, 128)

def random_point_from_set(s):
	return random.choice(tuple(s))