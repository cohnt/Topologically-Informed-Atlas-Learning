import numpy as np
from scipy.sparse import data
import scipy.spatial
import math
from sklearn.metrics import pairwise_distances
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import KernelPCA
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import KernelCenterer
from scipy.sparse.csgraph import connected_components
from src.util import edge_contraction, adj_mat_to_edge_set, edge_set_to_adj_mat, n_walk_closure

import src.visualization as visualization

class Atlas():
	# The Atlas class is the main class used for atlas learning. It handles initial chart creation,
	# chart combining, and computes the final embedding. The hole detection heuristic is passed in
	# as an argument, as is the manifold learning algorithm from sklearn that is used to compute
	# the embedding of each chart.
	def __init__(self, data, nbrs, target_dim, hole_detector, embed, viz_data=None):
		# data should be a (n,m) numpy array, where there are n samples in source dimension m.
		# nbrs should be a (n,n) neighborhood graph, where nbrs[i,j] is the distance between i
		# and j if they're connected, and 0 otherwise.
		# target_dim is the dimension of the manifold we're trying to learn.
		# hole_detector is a function that takes in a graph (represented by an adjacency matrix)
		# and outputs whether or not there's a (topological) hole in it. True means it has a hole,
		# False means it does not.
		# embed is an object from sklearn that actually computes the embedding from a precomputed
		# distance matrix. That is, one should be able to call embed.fit_transform(dist_mat). Note
		# that a distance matrix can be efficiently created from an adjacency matrix using the
		# graph_shortest_path function, from sklearn.utils.graph_shortest_path.
		# viz_data is a temporary thing I'm using with the temporary chart intersection
		# visualization. (Just ignore it!)

		self.data = data
		self.viz_data = viz_data if not (viz_data is None) else self.data
		self.n_data = self.data.shape[0]
		self.nbrs = nbrs
		self.shortest_dists = graph_shortest_path(self.nbrs)
		self.source_dim = self.data.shape[1]
		self.target_dim = target_dim
		self.hole_detector = hole_detector
		self.embed = embed
		self.atlas = dict()  # Created as a hashmap (dict), where the key is simply an integer that
		                     # identifies the chart, and the value is a set of indices, representing
		                     # the indices of points in self.data that are in the domain of the chart.
		self.internal_edges = dict() # This is identical to self.atlas, except the values are the
		                             # adjacency matrices of each chart, containing only edges
		                             # internal to those charts.
		self.atlas_overlap = None # This will be a graph on the set of coordinate charts, where two
		                          # charts have a connection if and only if they overlap
		self.maybe_can_combine = None # This will be a graph on the set of coordinate charts, where
		                              # two charts have a connection if and only if they haven't
		                              # already been checked for combining possibility, and failed.
		                              # If one of the two charts changes, all overlapping pairs are
		                              # restored.
		self.atlas_idx = [] # Mapping from index in the atlas_overlap graph to the atlas key.
		self.atlas_count = [] # Number of times each index in the atlas_overlap graph has been combined.
		self.embeddings = [] # The computed embeddings for each chart (w.r.t. the atlas_overlap graph).
		self.chart_errors = [] # The reconstruction error of each chart.
		self.chart_trustworthinesses = [] # The trustworthiness of each chart.
		self.chart_domains = [] # The domains of each chart, as returned by self.get_all_chart_domains
		self.inverses = [] # This stores a ChartInverseMapping object for each chart in the atlas. It can
		                   # be used to map points in a chart's embedding space back to the input space.
		self.embed_per_chart = [] # Each chart's unique Isomap object, ordered by atlas_idx
		self.non_overlap_charts = None # This will be identical to the self.atlas object, except it doesn't
		                               # undergo the overlapping expansion process. It's not used at all
		                               # within the Atlas class, but it can be used for cleaner visualizations.
		self.original_chart_count = None # Number of charts initially seeded.
		self.intersection_num_edges_list = [] # List of the number of edges checked in the intersection
		self.union_num_edges_list = [] # List of the number of edges checked in the union

	def seed_charts_pointwise(self, graph_radius):
		# This seeds the initial charts (i.e. fills in the self.atlas hashmap) by assigning a chart
		# to each point in the graph, and including their local neighborhood.
		# graph_radius should be a natural number, denoting the size of the local neighborhood. A
		# radius of 1 represents just the point and its neighbors, a radius of 2 includes the
		# neighbors of neighbors, and so on.

		# This is probably not the fastest way to do it, but the initial seeding of the charts isn't
		# where the main computational cost will be, so I'm not too worried about optimizing it.
		for i in range(self.n_data):
			# Each data point gets its own chart
			self.atlas[i] = set()
			self.atlas[i].add(i)
			self.internal_edges[i] = None
			self.atlas_idx.append(i)
			self.atlas_count.append(0)
		self.non_overlap_charts = self.atlas.copy()
		self._expand_charts(graph_radius)
		self._compute_atlas_overlap_graph()
		self._compute_chart_internal_edges()
		self.original_chart_count = len(self.atlas)

	def seed_charts_random(self, n_charts, overlap):
		# This seeds the initial charts (i.e. fills in the self.atlas hashmap) by randomly distributing
		# n_charts chart root points across the graph, and then assigning every point to the nearest chart.
		# overlap should be a natural number, denoting how far the graphs are expanded once they cover
		# the whole of the data. For overlap 1, the neighbors of every boundary point of the chart are
		# included, for overlap 2, the neighbors of the neighbors are added as well, and so on. Note 
		# that this leads to an effective doubling: the minimum width of the overlapping region will be
		# twice the value of overlap, since both neighboring charts expand into the other.

		# This is probably not the fastest way to do it, but the initial seeding of the charts isn't
		# where the main computational cost will be, so I'm not too worried about optimizing it.
		seed_idx = np.random.choice(self.n_data, n_charts)
		for i in range(n_charts):
			self.atlas[i] = set()
			self.atlas[i].add(seed_idx[i])
			self.internal_edges[i] = None
			self.atlas_idx.append(i)
			self.atlas_count.append(0)
		for i in range(self.n_data):
			dists = self.shortest_dists[i,seed_idx]
			best_idx = np.argmin(dists)
			self.atlas[best_idx].add(i)
		self.non_overlap_charts = self.atlas.copy()
		self._expand_charts(overlap)
		self._compute_atlas_overlap_graph()
		self._compute_chart_internal_edges()
		self.original_chart_count = len(self.atlas)

	def seed_charts_iterative_farthest_point(self, n_charts, overlap, metric=True):
		# This seeds the initial charts (i.e. fills in the self.atlas hashmap) by selecting a random
		# starting point, and then iteratively finding the farthest point from any already existing points.

		# overlap should be a natural number, denoting how far the graphs are expanded once they cover
		# the whole of the data. For overlap 1, the neighbors of every boundary point of the chart are
		# included, for overlap 2, the neighbors of the neighbors are added as well, and so on. Note 
		# that this leads to an effective doubling: the minimum width of the overlapping region will be
		# twice the value of overlap, since both neighboring charts expand into the other.

		# If metric is True, the "farthest" point is determined by approximate geodesic distance,
		# estimated by the shortest path along the neighborhood graph. This is useful for evenly
		# distributing charts across the manifold, even with uneven sampling.
		# If metric is False, then all edges are set to a uniform length beforehand. This is useful
		# for evenly distributing charts across the neighborhood graph, but may unevenly distribute
		# charts across the manifold as a whole.

		if metric:
			shortest_dists = self.shortest_dists
		else:
			uniform_adj = np.asarray(self.nbrs > 0, dtype=float)
			shortest_dists = graph_shortest_path(uniform_adj)

		# Select the farthest points in the graph
		# Implementation inspired by https://flothesof.github.io/farthest-neighbors.html
		seed_idx = np.zeros(n_charts, dtype=int) * np.random.randint(0, self.n_data)
		for s in range(1, n_charts):
			distances = [np.min(shortest_dists[i,seed_idx])
			             for i in range(0, self.n_data)]
			next_idx = np.argmax(distances)
			seed_idx[s] = next_idx

		# Now actually seed the charts
		for i in range(n_charts):
			self.atlas[i] = set()
			self.atlas[i].add(seed_idx[i])
			self.internal_edges[i] = None
			self.atlas_idx.append(i)
			self.atlas_count.append(0)
		for i in range(self.n_data):
			dists = shortest_dists[i,seed_idx]
			best_idx = np.argmin(dists)
			self.atlas[best_idx].add(i)
		self.non_overlap_charts = self.atlas.copy()
		self._expand_charts(overlap)
		self._compute_atlas_overlap_graph()
		self._compute_chart_internal_edges()
		self.original_chart_count = len(self.atlas)

	def _expand_charts(self, graph_radius):
		# This takes every chart in self.atlas, and iteratively adds its neighbors graph_radius times.
		for chart_idx, chart in self.atlas.items():
			chart_boundary = chart.copy()
			for j in range(graph_radius):
				# Recall: graph_radius is the number of times we have to expand how far we search
				chart = chart | chart_boundary # Add the current boundary points to the chart
				for k in chart_boundary:
					# For each point in the current boundary, add its neighbors to the current boundary
					local_nbhd = set(np.nonzero(self.nbrs[k])[0])
					chart_boundary = chart_boundary | local_nbhd
				# Remove any points from the current boundary that are already in the chart, so we don't
				# search over points we've already seen.
				chart_boundary = chart_boundary - chart
			chart = chart | chart_boundary # Need to do this one more time at the end
			self.atlas[chart_idx] = chart

	def _compute_atlas_overlap_graph(self):
		# This function is simply used to initially populate self.atlas_overlap after the initial
		# seeding of the coordinate charts. It creates an adjacency graph on n vertices, where n
		# is the number of charts. self.atlas_overlap[i,j] is 1 if the charts overlap, and 0 if not.
		# It also populates self.maybe_can_combine, as it's initially the same as self.atlas_overlap.
		atlas_len = len(self.atlas)
		self.atlas_overlap = np.zeros((atlas_len, atlas_len), dtype=bool)
		for i in range(atlas_len):
			for j in range(i+1, atlas_len):
				if bool(self.atlas[i] & self.atlas[j]):
					# self.atlas[i] & self.atlas[j] returns their intersection, and casting a set to a
					# boolean returns True if it is nonempty. So we only enter this if statement if
					# the two charts have nonempty intersection, i.e., they overlap
					self.atlas_overlap[i,j] = self.atlas_overlap[j,i] = 1
		self.maybe_can_combine = self.atlas_overlap.copy()

	def _domain_internal_edges(self, atlas_idx):
		# This function returns the edge set of a single coordinate domain.
		return adj_mat_to_edge_set(self.full_subgraph(np.array(list(self.atlas[atlas_idx]))))

	def _compute_chart_internal_edges(self):
		# This function is used to initially populate self.internal_edges. Before this function
		# is called, the key-value pairs in the dictionary are already set up -- this is handled
		# in the chart seeding function. But this function populates the value field with the
		# associated neighborhood graph as an adjacency matrix.
		for i in self.atlas_idx:
			self.internal_edges[i] = self._domain_internal_edges(i)

	def _combine_charts_no_check(self, i, j):
		# This function is UNSAFE! It just combines the charts, without checking whether or not
		# they can be combined. It's an internal function, that should only be called by other
		# methods. The point is to centralize the identical code, and let the wrapper functions
		# handle the questions of how and when an which, without duplicating a ton of code.

		if i > j:
			i, j = j, i # We always want to combine charts to lower indices
		# Combine the charts into i, delete j, then contract their edge in self.atlas_overlap
		self.atlas[self.atlas_idx[i]] = self.atlas[self.atlas_idx[i]] | self.atlas[self.atlas_idx[j]]
		self.atlas.pop(self.atlas_idx[j])
		self.non_overlap_charts[self.atlas_idx[i]] = self.non_overlap_charts[self.atlas_idx[i]] | self.non_overlap_charts[self.atlas_idx[j]]
		self.non_overlap_charts.pop(self.atlas_idx[j])
		self.internal_edges[self.atlas_idx[i]] = self.internal_edges[self.atlas_idx[i]] | self.internal_edges[self.atlas_idx[j]]
		# self.internal_edges[self.atlas_idx[i]] = self._domain_internal_edges(self.atlas_idx[i])
		self.internal_edges.pop(self.atlas_idx[j])
		self.atlas_overlap = edge_contraction(self.atlas_overlap, i, j)
		self.atlas_idx.pop(j)
		self.atlas_count[i] = self.atlas_count[i] + self.atlas_count[j] + 1
		self.atlas_count.pop(j)
		# Update self.maybe_can_combine by edge contracting, and then copying over the atlas_overlap
		# contents for the new chart to reset it, since we don't know if it can be combined anymore.
		self.maybe_can_combine = edge_contraction(self.maybe_can_combine, i, j)
		self.maybe_can_combine[i,:] = self.atlas_overlap[i,:]
		self.maybe_can_combine[:,i] = self.atlas_overlap[:,i]

	def _get_intersection(self, atlas_i, atlas_j):
		# Internal function used to get the subgraph of the neighborhood graph for only the vertices
		# in the intersection of the charts with keys atlas_i and atlas_j. Note that the scheme is
		# *not* the same as that used by self.atlas_overlap. Pass i and j into self.atlas_idx, and
		# then pass that into this function.
		intersection_idx = np.array(list(self.atlas[atlas_i] & self.atlas[atlas_j]))
		intersection_adj_mat = self.neighborhood_induced_subgraph(intersection_idx)
		# return intersection_adj_mat
		# print(self.internal_edges[atlas_i] & self.internal_edges[atlas_j])
		# print(intersection_idx)
		full_mat = edge_set_to_adj_mat(self.internal_edges[atlas_i] & self.internal_edges[atlas_j], self.n_data)
		# print(full_mat[np.ix_(intersection_idx, intersection_idx)])
		# print(intersection_adj_mat)
		# exit(0)

		# if len(self.atlas) < np.inf:
		# 	import matplotlib.pyplot as plt
		# 	from mpl_toolkits import mplot3d
		# 	fig = plt.figure()
		# 	ax = fig.add_subplot(1, 1, 1, projection="3d")
		# 	dom_1 = np.array(list(self.atlas[atlas_i] - self.atlas[atlas_j]))
		# 	dom_2 = np.array(list(self.atlas[atlas_j] - self.atlas[atlas_i]))
		# 	if len(dom_1) > 0:
		# 		ax.scatter(self.viz_data[dom_1,0], self.viz_data[dom_1,1], self.viz_data[dom_1,2], c="red")
		# 		# visualization.draw_neighbors(ax, self.viz_data[dom_1], edge_set_to_adj_mat(self.internal_edges[atlas_i], self.n_data)[np.ix_(dom_1, dom_1)], c="red")
		# 		# visualization.draw_neighbors(ax, self.data[np.array(list(self.atlas[atlas_i]))], self.nbrs[np.ix_(np.array(list(self.atlas[atlas_i])), np.array(list(self.atlas[atlas_i])))], c="red")
		# 	if len(dom_2) > 0:
		# 		ax.scatter(self.viz_data[dom_2,0], self.viz_data[dom_2,1], self.viz_data[dom_2,2], c="blue")
		# 		# visualization.draw_neighbors(ax, self.viz_data[dom_2], edge_set_to_adj_mat(self.internal_edges[atlas_j], self.n_data)[np.ix_(dom_2, dom_2)], c="blue")
		# 		# visualization.draw_neighbors(ax, self.data[np.array(list(self.atlas[atlas_j]))], self.nbrs[np.ix_(np.array(list(self.atlas[atlas_j])), np.array(list(self.atlas[atlas_j])))], c="blue")
		# 	if len(intersection_idx) > 0:
		# 		ax.scatter(self.viz_data[intersection_idx,0], self.viz_data[intersection_idx,1], self.viz_data[intersection_idx,2], c="green")
		# 		# visualization.draw_neighbors(ax, self.viz_data[intersection_idx], intersection_adj_mat, c="green")
		# 	plt.show()

		return full_mat[np.ix_(intersection_idx, intersection_idx)]

	def neighborhood_induced_subgraph(self, idx):
		# Returns the induced subgraph of self.nbrs by the indices idx.
		return self.nbrs[np.ix_(idx, idx)]

	def full_subgraph(self, idx):
		# Returns the induced subgraph of self.nbrs by the indices idx, but keeping the original shape.
		temp = np.zeros(self.nbrs.shape, dtype=int)
		temp[np.ix_(idx, idx)] = np.ceil(self.nbrs[np.ix_(idx, idx)]).astype(int)
		return temp

	def combine_charts_random(self):
		# For now, just pick two charts and try to combine them.

		# The first chart, i, is just randomly chosen
		# The second chart, j is randomly chosen from the charts we know overlap with the first chart
		i = np.random.choice(len(self.atlas))
		nonzero = np.nonzero(self.atlas_overlap[i])[0]
		j_idx = np.random.choice(len(nonzero))
		j = nonzero[j_idx]

		if not self.maybe_can_combine[i,j]:
			# We've tried this combination before, so we know it'll fail
			print("Skip!")
			return False

		# Compute the neighborhood subgraph of the intersection of the charts
		intersection_adj_mat = self._get_intersection(self.atlas_idx[i], self.atlas_idx[j])

		intersection_num_edges = np.count_nonzero(intersection_adj_mat) / 2
		union_num_edges = len(self.internal_edges[self.atlas_idx[i]] | self.internal_edges[self.atlas_idx[j]])
		self.intersection_num_edges_list.append(intersection_num_edges)
		self.union_num_edges_list.append(union_num_edges)

		# The intersection must be connected and have no holes in order to safely combine the charts
		if connected_components(intersection_adj_mat)[0] == 1:
			if not self.hole_detector(intersection_adj_mat):
				self._combine_charts_no_check(i, j)
				return True
			else:
				print("Hole detected.")
				self.maybe_can_combine[i,j] = self.maybe_can_combine[j,i] = 0
				return False
		else:
			print("Charts not connected.")
			self.maybe_can_combine[i,j] = self.maybe_can_combine[j,i] = 0
			return False

	def combine_charts_minimum(self, both=False):
		# Pick the chart which has been combined the fewest times, and try to combine with that. If
		# both is True, it picks the adjacent chart with the fewest combines; if both is False, it
		# just chooses randomly.

		i = np.argmin(self.atlas_count)
		nonzero = np.nonzero(self.atlas_overlap[i])[0]
		if both:
			j_idx = np.argmin(np.asarray(self.atlas_count)[nonzero])
			j = nonzero[j_idx]
		else:
			j_idx = np.random.choice(len(nonzero))
			j = nonzero[j_idx]

		if not self.maybe_can_combine[i,j]:
			# We've tried this combination before, so we know it'll fail
			print("Skip!")
			return False

		# Compute the neighborhood subgraph of the intersection of the charts
		intersection_adj_mat = self._get_intersection(self.atlas_idx[i], self.atlas_idx[j])

		intersection_num_edges = np.count_nonzero(intersection_adj_mat) / 2
		union_num_edges = len(self.internal_edges[self.atlas_idx[i]] | self.internal_edges[self.atlas_idx[j]])
		self.intersection_num_edges_list.append(intersection_num_edges)
		self.union_num_edges_list.append(union_num_edges)

		# The intersection must be connected and have no holes in order to safely combine the charts
		if connected_components(intersection_adj_mat)[0] == 1:
			if not self.hole_detector(intersection_adj_mat):
				self._combine_charts_no_check(i, j)
				return True
			else:
				print("Hole detected.")
				self.maybe_can_combine[i,j] = self.maybe_can_combine[j,i] = 0
				return False
		else:
			print("Charts not connected.")
			self.maybe_can_combine[i,j] = self.maybe_can_combine[j,i] = 0
			return False

	def combine_charts_exhaustive(self, ordered=False):
		# Try to combine charts until either one has been found, or every combination has been
		# tried. If ordered is True, then it tries in lexicographical order by index. But if
		# ordered is False, it shuffle this order.

		# Get the nonzero indices from only the upper triangular portion, since we don't want
		# to check every pair of charts for overlap twice -- (i,j) and (j,i) are effectively
		# the same.
		nonzero_idx = np.nonzero(np.triu(self.atlas_overlap))
		overlapping_pairs = np.array(nonzero_idx).T # Structure as a list of pairs of charts
		if not ordered:
			np.random.shuffle(overlapping_pairs)
			# np.random.shuffle acts on only the first axis by default

		# Now, we try to combine the charts until we succeed or run out
		for i, j in overlapping_pairs:
			if not self.maybe_can_combine[i,j]:
				# We've tried this combination before, so we know it'll fail
				continue

			# Compute the neighborhood subgraph of the intersection of the charts
			intersection_adj_mat = self._get_intersection(self.atlas_idx[i], self.atlas_idx[j])

			intersection_num_edges = np.count_nonzero(intersection_adj_mat) / 2
			union_num_edges = len(self.internal_edges[self.atlas_idx[i]] | self.internal_edges[self.atlas_idx[j]])
			self.intersection_num_edges_list.append(intersection_num_edges)
			self.union_num_edges_list.append(union_num_edges)

			# The intersection must be connected and have no holes in order to safely combine the charts
			if connected_components(intersection_adj_mat)[0] == 1:
				if not self.hole_detector(intersection_adj_mat):
					self._combine_charts_no_check(i, j)
					return True
				else:
					print("Not Combining: Found a hole!")
					self.maybe_can_combine[i,j] = self.maybe_can_combine[j,i] = 0
			else:
				print("Not Combining: Intersection not connected!")
				self.maybe_can_combine[i,j] = self.maybe_can_combine[j,i] = 0

		# If all of the pairs of charts have failed, then
		return False

	def get_chart_names(self):
		# Returns a list of the keys from the atlas. This can be useful for interfacing with some of
		# the later methods, such as get_chart_domain, or for directly interacting with self.atlas
		return list(self.atlas.keys())

	def get_chart_domain(self, atlas_i, ptcc=0):
		# This returns a single coordinate chart, in terms of its domain in the manifold. It
		# returns a list of data points contained in the chart and the induced neighborhood subgraph.
		# ptcc is the partial transitive closure count -- how many times to add transitive edges
		# to the graph. Higher values take longer to compute, especially for large graphs, but
		# it can help add back removed edges that should be included.
		idx = np.array(list(self.atlas[atlas_i]))
		data = self.data[idx]

		# Only grab the internal edges of the domain. This is very important, as the chart
		# may contain all the vertices in a loop, but without the edge that would close it.
		edges_adj = edge_set_to_adj_mat(self.internal_edges[atlas_i], self.n_data)[np.ix_(idx, idx)]
		edges_adj = n_walk_closure(edges_adj, ptcc)
		edges = np.multiply(edges_adj, self.nbrs[np.ix_(idx, idx)])
		return data, edges

	def get_all_chart_domains(self, ptcc=0):
		# This returns all of the coordinate charts, in terms of their domains in the manifold.
		# It does so by returning a list of tuples (points, edges), where points is the list of
		# points contained in the chart, and edges is the induced neighborhood subgraph.
		# See the get_chart_domain method for an explation of ptcc.
		self.chart_domains = [self.get_chart_domain(atlas_i, ptcc=ptcc) for atlas_i in self.atlas_idx]
		return self.chart_domains

	def embed_charts(self, ptcc=0):
		# Embed each chart in self.atlas, and populate them into self.embeddings.
		# Save each Isomap object into self.embed_per_chart.
		# See the get_chart_domain method for an explation of ptcc.
		if len(self.chart_domains) == 0:
			# This means the domains haven't been computed yet
			self.get_all_chart_domains(ptcc=ptcc)
		for domain in self.chart_domains:
			cloned = clone(self.embed)
			cloned.fit(graph_shortest_path(domain[1]))
			self.embed_per_chart.append(cloned)
			self.embeddings.append(cloned.transform(graph_shortest_path(domain[1])))
			self.chart_errors.append(cloned.reconstruction_error())
		self._compute_chart_trustworthiness()

	def embed_charts_kpca(self, ptcc=0):
		# Alternative method of embedding charts using KernelPCA.
		# Embed each chart in self.atlas, and populate them into self.embeddings.
		# Save each KernelPCA object into self.embed_per_chart.
		# See the get_chart_domain method for an explanation of ptcc.
		if len(self.chart_domains) == 0:
			# This means the domains haven't been computed yet
			self.get_all_chart_domains(ptcc=ptcc)
		for domain in self.chart_domains:
			# Embed with kernel PCA
			kpca = KernelPCA(n_components=self.target_dim, kernel="precomputed", n_jobs=-1)
			shortest_distances = graph_shortest_path(domain[1])
			kmat = -0.5 * (shortest_distances**2)
			embedding = kpca.fit_transform(kmat)

			# Save relevant variables
			self.embed_per_chart.append(kpca)
			self.embeddings.append(embedding)
		self._compute_chart_trustworthiness()
		self._compute_kpca_reconstruction_error()

	def construct_inverse_mapping(self):
		for i in range(len(self.atlas_idx)):
			atlas_i = self.atlas_idx[i]
			idx = np.array(list(self.atlas[atlas_i]))
			data = self.data[idx] # Get the input data
			embedding = self.embeddings[i] # Get the embedded data
			self.inverses.append(ChartInverseMapping(data, embedding))

	def nn_classification(self, test_data, train_labels):
		"""
		Given data, run nearest neighbors classification using
		the generated embeddings.
		This function checks all charts within the atlas to see which
		chart contains the *nearest* neighbor in Isomap's embedding 
		space, and uses the label of that nearest overall point.
		Inputs:
			- test_data = (N, D) test data
			- train_labels = (N,) array of labels
		"""
		# Multi points
		nearest_idxs = np.ones(len(test_data)) * -1
		test_dist_mat = pairwise_distances(test_data, self.data)
		min_dists = np.asarray([math.inf] * len(test_dist_mat))
		for chart_idx in range(len(self.embed_per_chart)):
			embedder = self.embed_per_chart[chart_idx]			
			dists, _ = embedder.nbrs_.kneighbors(test_dist_mat, n_neighbors=1)
			change_idxs = np.argwhere(dists[...,0] < min_dists)
			min_dists[change_idxs] = dists[change_idxs][..., 0]
			data_idxs = np.array(list(self.atlas[chart_idx])) # indices of full dataset
			nearest_idxs = data_idxs[change_idxs]
		
		return train_labels[nearest_idxs]

	def transition_map(self, chart_idx, point):
		# Checks if a point in an embedding chart_idx is within an overlapping region
		# with another chart, and if so, returns its location in those charts.
		
		# The inverse mapping is needed for this functionality
		if len(self.inverses) == 0:
			self.construct_inverse_mapping()

		# Get the simplex and barycentric coordinates
		simplex_indices, convex_comb = self.inverses[chart_idx].get_simplex_weights(point)
		global_indices = list(self.atlas[self.atlas_idx[chart_idx]])
		global_simplex_indices = [global_indices[i] for i in simplex_indices]
		idx_set = set(global_simplex_indices)

		# Check each chart domain to see if the simplex is contained in it
		chart_locations = []
		for i in range(len(self.atlas_idx)):
			a_idx = self.atlas_idx[i]
			if idx_set.issubset(self.atlas[a_idx]):
				# local_simplex_indices = [list(self.atlas[i])[j] for j in global_simplex_indices]
				local_simplex_indices = np.where(np.in1d(np.array(list(self.atlas[a_idx])), np.array(global_simplex_indices)))[0]
				chart_locations.append(self.inverses[i].apply_simplex_weights(local_simplex_indices, convex_comb, self.embeddings[i]))
			else:
				chart_locations.append([])

		return chart_locations

	def _compute_kpca_reconstruction_error(self):
		# Populates self.chart_errors with the ISOMAP reconstruction error,
		# as computed directly from the KPCA object used in the embeddings.
		# https://github.com/scikit-learn/scikit-learn/blob/2beed5584/sklearn/manifold/_isomap.py#L171
		for i in range(len(self.embeddings)):
			kpca = self.embed_per_chart[i]
			dist_mat = graph_shortest_path(self.chart_domains[i][1])
			G = -0.5 * dist_mat ** 2
			G_center = KernelCenterer().fit_transform(G)
			evals = kpca.lambdas_
			err = np.sqrt(np.sum(G_center ** 2) - np.sum(evals ** 2)) / G.shape[0]
			self.chart_errors.append(err)

	def _compute_chart_trustworthiness(self):
		# Populates self.chart_errors with the embedding trustworthiness
		for i in range(len(self.embeddings)):
			orig = self.chart_domains[i][0]
			embed = self.embeddings[i]
			self.chart_trustworthinesses.append(trustworthiness(orig, embed))

class ChartInverseMapping():
	# This is a wrapper for the Delaunay triangulations, that's simply used to map points from
	# the embedding space to the input space.
	def __init__(self, source_data, target_data):
		# source_data should be the points in the input space, with shape (n,src_dim)
		# target_data should be the corresponding embedded points, with shape (n,target_dim)
		self.source_data = source_data
		self.target_data = target_data
		self.source_dim = self.source_data.shape[1]
		self.target_dim = self.target_data.shape[1]

		if self.target_dim == 1:
			self.tri = Delaunay1D(self.target_data)
		else:
			self.tri = scipy.spatial.Delaunay(self.target_data, qhull_options="QJ")
			# "QJ" ensures that all points are used in the triangulation, even if they're coplanar.

	def inverse_mapping(self, points):
		# points should be a numpy array of shape [target_dim] or [*,target_dim]
		# This function will return a numpy array of shape [source_dim] or [*,source_dim]
		if points.ndim == 1:
			return self.single_inverse_mapping(points)
		elif points.ndim == 2:
			mapped_points = np.zeros((len(points), self.source_dim))
			for i in range(len(points)):
				mapped_points[i,:] = self.single_inverse_mapping(points[i])
			return mapped_points

	def get_simplex_weights(self, point):
		# Returns the simplex indices, and the barycentric coordinates
		simplex_num = self.tri.find_simplex(point)
		if simplex_num == -1:
			print("Error: coordinate outside of convex hull!")
			print(point)
			raise ValueError

		# Grab the points on the vertices of the simplex
		simplex_indices = self.tri.simplices[simplex_num]
		simplex = self.tri.points[simplex_indices]

		# Write as convex combination of simplex vertices
		A = np.vstack((simplex.T, np.ones((1, self.target_dim+1))))
		b = np.vstack((point.reshape(-1, 1), np.ones((1, 1))))
		convex_comb = np.linalg.solve(A, b)
		convex_comb = np.asarray(convex_comb).flatten()

		new_idx = np.argsort(simplex_indices)

		return simplex_indices[new_idx], convex_comb[new_idx]

	def apply_simplex_weights(self, simplex_indices, convex_comb, data):
		factors = np.zeros(len(data))
		factors[simplex_indices] = convex_comb
		factors_mat = np.diag(factors)
		mapped_point = np.sum(np.matmul(factors_mat, data), axis=0).flatten()

		return mapped_point

	def single_inverse_mapping(self, point):
		simplex_indices, convex_comb = self.get_simplex_weights(point)
		return self.apply_simplex_weights(simplex_indices, convex_comb, self.source_data)

	def check_domain(self, points):
		if points.ndim == 1:
			return (self.tri.find_simplex(points) != -1)
		elif points.ndim == 2:
			simplex_nums = self.tri.find_simplex(points)
			return simplex_nums != -1

class Delaunay1D():
	# This is used just like scipy.spatial.Delaunay, but for 1-manifolds, since the scipy version
	# requires a minimum dimension of 2. (Taken from my deformable object manifold learning project.)
	def __init__(self, data):
		self.points = data
		self.points_flat = self.points.flatten()
		self.inds = np.argsort(self.points_flat)
		self.sorted_points = self.points_flat[self.inds]
		self.simplices = np.array([self.inds[:-1], self.inds[1:]]).T

	def find_simplex(self, point):
		idx = np.searchsorted(self.sorted_points, point)
		if idx == 0 or idx == len(self.points):
			return -1
		simplex_idx = self.inds[idx][0]
		return simplex_idx
