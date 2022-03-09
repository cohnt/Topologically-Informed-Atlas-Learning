import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.patches import Polygon

hash_max = 2.0 ** sys.hash_info.hash_bits

def draw_neighbors(ax, points, adj_mat, **kwargs):
	# Draws the neighborhood connections onto a given axis
	# adj_mat should be an adjacency matrix. Both
	# connectivity and distance are acceptable. points should
	# be a nx2 or nx3 list of 2d or 3d points.
	dim = points.shape[1]
	for i in range(adj_mat.shape[0]):
		for j in range(i, adj_mat.shape[1]):
			if adj_mat[i,j] > 0:
				line = np.array([points[i], points[j]])
				if dim == 2:
					ax.plot(line[:,0], line[:,1], **kwargs)
				elif dim >= 3:
					ax.plot(line[:,0], line[:,1], line[:,2], **kwargs)

def draw_cycle(ax, points, cycle, **kwargs):
	# Draws a cycle onto a given axis. cycle should be a
	# list of indices (where the last index is implicitly
	# connected to the first, although it's not an issue
	# if it's explicitly included). points should be a
	# nx2 or nx3 list of 2d or 3d points.
	dim = points.shape[1]
	line = np.array([points[idx] for idx in cycle])
	if dim == 2:
		ax.plot(line[:,0], line[:,1], **kwargs)
		ax.plot(line[[0,-1],0], line[[0,-1],1], **kwargs)
	elif dim >= 3:
		ax.plot(line[:,0], line[:,1], line[:,2], **kwargs)
		ax.plot(line[[0,-1],0], line[[0,-1],1], line[[0,-1],2], **kwargs)

def draw_atlas_domains(ax, chart_domains, chart_labels, **kwargs):
	# Draws all of the points in the atlas, colored by which chart they're in

	# Stack all of the chart domains into one giant list of points
	# Then take only the unique points in that list
	points, idx = np.unique(np.vstack([chart_domains[i][0] for i in range(len(chart_domains))]), axis=0, return_index=True)

	# Label each point in the stack by the chart's index
	domain_labels = np.hstack([np.ones(chart_domains[i][0].shape[0])*float(hash(chart_labels[i]) / hash_max) for i in range(len(chart_domains))])[idx]

	dim = points.shape[1]
	if dim == 2:
		ax.scatter(points[:,0], points[:,1], c=domain_labels, cmap=plt.get_cmap("tab20"), **kwargs)
	elif dim >= 3:
		ax.scatter(points[:,0], points[:,1], points[:,2], c=domain_labels, cmap=plt.get_cmap("tab20"), **kwargs)

def draw_non_overlap_domains(ax, data, domains_obj, init_num_charts, **kwargs):
	# Draws all of the points in the atlas, covered by which non-overlapping chart they're in
	# points should be the list of all points being displayed
	# data should be a dictionary, like in atlas.non_overlap_charts

	points = []
	labels = []
	for chart_id, domain in domains_obj.items():
		for idx in domain:
			points.append(data[idx])
			labels.append(chart_id)
	points = np.array(points)
	labels = np.array(labels)

	dim = data.shape[1]
	if dim == 2:
		ax.scatter(points[:,0], points[:,1], c=labels, cmap=plt.get_cmap("prism", vmin=0, vmax=init_num_charts), **kwargs)
	elif dim >= 3:
		ax.scatter(points[:,0], points[:,1], points[:,2], c=labels, cmap=plt.get_cmap("prism"), vmin=0, vmax=init_num_charts, **kwargs)

def draw_triangles(ax, points, tri, point_colors, edge_color=None, max_side_len=np.inf, cmap=None, **kwargs):
	# Draws a 2D triangulation. points is the point set, and tri is
	# the Delaunay triangulation (a scipy.spatial.Delaunay object).
	# For coloring, point_colors may either be a 1D array of colormap
	# values or a single color (given as a string). edge_color can
	# be used to specify a color of the edges; if it's none, edges
	# aren't displayed; if it's auto, edges are the same as the face
	# color. max_side_len can be set so any triangle with a side
	# longer than this threshold isn't displayed; the default, np.inf,
	# means all triangles will be shown.

	# This function works for both 2D and 3D plots, so that triangles
	# can be drawn in the embedding space, and the source space if the
	# ambient space is 3D. (This is nice for the synthetic examples.)

	if not isinstance(point_colors, str):
		if cmap is None:
			print("If point_colors is a list of colormap inputs, the cmap must be specified.")
			print("For example, cmap=plt.cm.Spectral")
			exit(1)

	dim = points.shape[1]

	for simplex in tri.simplices:
		p = points[simplex]
		if max_side_len < np.inf:
			if np.linalg.norm(p[0] - p[1]) > max_side_len or \
			   np.linalg.norm(p[1] - p[2]) > max_side_len or \
			   np.linalg.norm(p[2] - p[0]) > max_side_len:
				# Can't draw this triangle
				continue

		# Compute the face color
		if isinstance(point_colors, str):
			fc = point_colors
		else:
			# Get the mean color of each point in the triangle
			fc = np.mean(cmap(point_colors[simplex]), axis=0)

		# Compute the edge color
		if edge_color is None:
			ec = None
		elif edge_color == "auto":
			ec = fc
		else:
			ec = edge_color

		# Draw the triangle
		if dim == 2:
			ax.add_patch(Polygon(p, closed=True, edgecolor=ec, facecolor=fc, **kwargs))
		elif dim >= 3:
			ax.plot_trisurf(p[:,0], p[:,1], p[:,2], edgecolor=ec, color=fc, shade=False, **kwargs)