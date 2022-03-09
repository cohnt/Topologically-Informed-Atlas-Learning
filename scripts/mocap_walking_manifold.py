#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from os.path import isdir
from os import makedirs
from matplotlib import offsetbox
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph, KNeighborsTransformer, NearestNeighbors
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.metrics import mean_squared_error as mse_loss
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap, trustworthiness

import mocap.datasets.cmu as CMU
import mocap.processing.normalize as norm
import mocap.visualization.humanpose as hviz
from mocap.visualization.sequence import SequenceVisualizer

import src.visualization as visualization
import src.atlas as atlas
import src.cyclecut as cyclecut
import src.geometry as geometry
import src.util as util

print("You are visualizing an atlas embedding of a walking example from the CMU Mocap dataset.")
print("Please allow some time for the embedding to generate.")

def get_adj_mat(flat, neighbors_k):
	"""
	Inputs:
		- flat = (N, D)-sized flattened input digit matrix
		- neighbors_k = number of neighbors for which to perform kNN
	Outputs:
		- adj_mat = (N, N) adjacency matrix with edges between data points
	"""
	adj_mat = KNeighborsTransformer(mode="distance", n_neighbors=neighbors_k, n_jobs=-1).fit_transform(flat).toarray()
	adj_mat = np.maximum(adj_mat, adj_mat.T)
	return adj_mat

def create_atlas(flat, adj_mat):
	"""
	Inputs:
		flat = (N, D)-sized flattened input digit matrix
		adj_mat = (D, D)-sized adjacency matrix
	Outputs:
		a = instance of Atlas class
	"""
	L = 15
	def hole_detector(adj_mat):
		return len(cyclecut.find_large_atomic_cycle(adj_mat, L)) > 0

	a = atlas.Atlas(flat, 
					adj_mat, 
					1, 
					hole_detector, 
					Isomap(n_components=2, metric="precomputed"))
	a.seed_charts_iterative_farthest_point(25, 2) # n_charts, overlap

	while len(a.atlas) > 1:
		if not a.combine_charts_exhaustive():
			break
		# print(len(a.atlas)) # (int)

	a.embed_charts()
	# print(a.chart_errors) # ([err, err])

	return a

def plot_embedding_isomap(X, ax, param, title=None):
	"""
	Plots isomap embedding.
	Inputs:
		- X: data
		- ax: mpl subplot axis to plot on
		- title: figure title
	"""
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)

	ax.scatter(X[:,0], X[:,1], c=param, cmap=plt.get_cmap("hsv"))

	plt.xticks([]), plt.yticks([])
	if title is not None:
		plt.title(title)

def unit_vector(vector):
	""" Returns the normalized unit vector.  """
	return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
	""" 
	Returns angle between two vectors (radians, 0 to 2pi)
	"""
	v = v2 - v1	
	return np.arctan2(v[1], v[0])

def compute_parametrization_angles(isomap_embeddings):
	"""
	Gets angles of isomap embedded points relative to
	isomap centroid and returns them for parameterization.
	"""
	centroid = np.mean(isomap_embeddings, axis=0)
	n_samples = isomap_embeddings.shape[0]
	angles = np.zeros((n_samples))
	for i in range(n_samples):
		row = isomap_embeddings[i]
		angles[i] = angle_between(centroid, row)
	return angles

def compute_parametrization_loopy(X):
	"""
	Parameterizes atlas embedded data to plot in a half-circle.
	"""
	# normalize the points in X to lie on [0,1]
	# plot (cos pi t, sin pi t)
	t = X[:,0] / np.max(X[:,0])
	return np.cos(np.pi * t), np.sin(np.pi * t)

def plot_embedding_atlas(X, angles, ax, title=None, period=1):
	"""
	Inputs:
		- X: data
		- angles: isomap angles
		- ax: subplot axis to plot on
	"""
	x_min, x_max = np.min(X, 0), np.max(X, 0)
	X = (X - x_min) / (x_max - x_min)
	X = X[::period]

	max_norm_x = np.max(X[:, 0])
	xs, ys = compute_parametrization_loopy(X)
	ax.scatter(xs, ys, c=angles[::period], cmap=plt.get_cmap("hsv"))

	plt.xticks([]), plt.yticks([])
	if title is not None:
		ax.set_title(title)

	return x_min, x_max, max_norm_x


def plot_source_poses(atlas, k, vis, idx, raw_shape, create_video=False, noaxis=True, views=[(0,90)]):
	# use vis to plot source pose for idxth frame in kth coordinate chart in atlas
	# hacky solution since there is no structure tracking source data idxs from embeddings
	atlas.construct_inverse_mapping()
	inverse_mappings = atlas.inverses
	vis.plot(inverse_mappings[k].source_data[idx].reshape(raw_shape),
			  parallel=False,
			  plot_jid=False,
			  noaxis=noaxis,
			  create_video=create_video,
			  views=views)

def get_subj_data(subjs=['02']):
	# subjs = ['02', '02', '06', '09', '13', '14', '29', '35', '49', '55', '125']
	# actions = ['01', '03', '02', '03', '30', '34', '24', '24', '08', '01', '04']
	# actions: walk, run, dribble forward, run, jumpingjacks++, climb ladder, snake imitation, run/jog, cartwheels, dance/whirl, breast stroke`    
	ds = CMU.CMU(subjs)
	return ds

print("Please drag your mouse through either of the top two charts in the interactive figure.")
print("You will notice the crosshairs moving through both bottom charts when the embedded point lies in the overlap.")
print("The walking figure at the bottom will change as you move along the embedding space.")
print("Note that error messages may display as you move through the embedding. You may ignore them; they do not contribute to the functionality or visualization.")

while True:
	proceed = input("Proceed? (y/n)")

	if proceed.lower().strip() == 'y':
		break
	elif proceed.lower().strip() == 'n':
		exit()
	else:
		print("Not a valid input. Please enter y or n.")

vis_dir = '../'
if not isdir(vis_dir):
	makedirs(vis_dir)

create_video = False
noaxis = True
to_file = True
mark_origin = False

period = 50

views = [(0, 60)]

vis = SequenceVisualizer(vis_dir, 'vis_cmu',
							to_file=to_file,
							mark_origin=mark_origin)

ds = get_subj_data()
action_sample = ds[0]
frame_nums = range(len(action_sample))
action_sample = action_sample - np.expand_dims(action_sample[:,0,:],axis=1)

n_samples = action_sample.shape[0]
neighbors_k = 16
flat = action_sample.reshape((n_samples, -1))
ism = Isomap(n_neighbors=neighbors_k, n_components=2)
isomap_embeddings = ism.fit_transform(flat)
adj_mat = get_adj_mat(flat, neighbors_k)
a = create_atlas(flat, adj_mat)

print("Chart errors: %s" % a.chart_errors)
print("Chart trustworthinesses: %s" % a.chart_trustworthinesses)
print("Mean trustworthiness: %s" % np.mean(a.chart_trustworthinesses))
print("Min trustworthiness: %s" % np.min(a.chart_trustworthinesses))

raw_embedding = Isomap(n_neighbors=neighbors_k, n_components=1).fit_transform(flat)
print("Isomap error: %s" % ism.reconstruction_error())
print("Isomap trustworthiness: %s" % trustworthiness(flat, isomap_embeddings))

# Autoencoder comparison
# data = flat.copy()
# import src.comparisons.autoencoder_atlas as atlas
# target_dim = 1
# n_charts = 4
# at = atlas.Atlas(target_dim, n_charts)
# at.fit(data)
# probs = at.chart_probs(data)
# chart_assignments = probs.argmax(axis=1)
# from keras import backend as K
# embs = at.encoder(K.constant(data))
# np_embs = [K.eval(emb) for emb in embs]
# chart_embs = [np_embs[i][chart_assignments == i,:] for i in range(n_charts)]
# trusts = []
# for i in range(len(chart_embs)):
# 	emb = chart_embs[i]
# 	if len(emb) == 0:
# 		continue
# 	orig = data[chart_assignments==i,:]
# 	trusts.append(trustworthiness(orig, emb))
# print("Chart trustworthinesses: %s" % trusts)
# print("Mean trustworthiness: %s" % np.mean(trusts))
# print("Min trustworthiness: %s" % np.min(trusts))

# Plot Isomap, compute vmin and vmax from isomap angles
angles = compute_parametrization_angles(isomap_embeddings)
vmin, vmax = np.min(angles), np.max(angles)

# fig = plt.figure()
# ax = plt.subplot(111)
# plot_embedding_isomap(isomap_embeddings, ax, angles)

############## PLOT CHARTS ###############
seed = 43
np.random.seed(seed)

a.construct_inverse_mapping()
inverse_mappings = a.inverses
fig = plt.figure()
# util.maximize_plt_fig(fig)
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5, projection=Axes3D.name)
ax5.axis('off')

color_idx = 0
save_count = 0
nearest_frame_num = 0

chart_idxs = [idx for idx in a.atlas]
embed_0 = a.embeddings[0]
embed_1 = a.embeddings[1]

ax1.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=angles[list(a.atlas[chart_idxs[0]])])
ax2.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=angles[list(a.atlas[chart_idxs[1]])])

x_min_0, x_max_0, max_norm_0 = plot_embedding_atlas(embed_0, angles[list(a.atlas[chart_idxs[0]])], ax3)
x_min_1, x_max_1, max_norm_1 = plot_embedding_atlas(embed_1, angles[list(a.atlas[chart_idxs[1]])], ax4)

def display_state(xy, embed_idx):
	global color_idx, nearest_frame_num

	if embed_idx == 0:
		maxnorm = max_norm_0
		xmin = x_min_0
		xmax = x_max_0
		xmin_other = x_min_1
		xmax_other = x_max_1
		maxnorm_other = max_norm_1
	elif embed_idx == 1:
		maxnorm = max_norm_1
		xmin = x_min_1
		xmax = x_max_1
		xmin_other = x_min_0
		xmax_other = x_max_0
		maxnorm_other = max_norm_0
	try:
		orig_point = a.inverses[embed_idx].single_inverse_mapping(xy)
		
		# non_param_x = (np.arccos(xy[0]) / np.pi) * maxnorm * (xmax-xmin) + xmin
		# non_param_y = (np.arcsin(xy[1]) / np.pi) * maxnorm * (xmax-xmin) + xmin
		non_param_xy = xy #np.array([non_param_x, non_param_y])
		transition = a.transition_map(embed_idx, non_param_xy)
		# if len(transition[1-embed_idx]) > 0:
		# 	other_transition = transition[1-embed_idx]
		# 	other_transition = (other_transition - xmin_other) / (xmax_other - xmin_other)

		# 	xs, ys = compute_parametrization_loopy(other_transition)
		# 	transition[1-embed_idx] = np.array([xs, ys])
		
	except ValueError:
		print("Inverse mapping error!")
		return

	idxs, weights = a.inverses[embed_idx].get_simplex_weights(xy)
	best_local_idx = np.argmax(weights)
	idx = idxs[best_local_idx]
	global_idx = list(a.atlas[a.atlas_idx[embed_idx]])[idx]
	nearest_frame_num = frame_nums[global_idx]

	ax1.cla()
	ax2.cla()
	ax1.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=angles[list(a.atlas[chart_idxs[0]])])
	ax2.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=angles[list(a.atlas[chart_idxs[1]])])
	ax3.cla()
	ax4.cla()
	plot_embedding_atlas(embed_0, angles[list(a.atlas[chart_idxs[0]])], ax3)
	plot_embedding_atlas(embed_1, angles[list(a.atlas[chart_idxs[1]])], ax4)

	if len(transition[0]) > 0:
		ax1.scatter([transition[0][0]], [transition[0][1]], color="black", marker="+", s=25**2)
	if len(transition[1]) > 0:
		ax2.scatter([transition[1][0]], [transition[1][1]], color="black", marker="+", s=25**2)
	
	# map markers (transition points) to the circle graphs
	if len(transition[embed_idx]) > 0:
		ct = transition[embed_idx]
		ct = (ct - xmin) / (xmax - xmin)	
		t = ct[0] / maxnorm
		xs, ys = np.cos(np.pi * t), np.sin(np.pi * t)
		ct = [xs, ys]
		ax3.scatter(ct[0], ct[1], color="black", marker="+", s=25**2)
		
	if len(transition[1-embed_idx]) > 0:
		ct_other = transition[1-embed_idx]
		ct_other = (ct_other - xmin_other) / (xmax_other - xmin_other)
		t_other = ct_other[0] / maxnorm_other
		xs, ys = np.cos(np.pi * t_other), np.sin(np.pi * t_other)
		ct_other = [xs, ys]
		ax4.scatter(ct_other[0], ct_other[1], color="black", marker="+", s=25**2)

	# ax3
	ax5.cla()
	ax5.axis('off')
	ax5.view_init(elev=views[0][0], azim=views[0][1])
	# ax3.axes.set_aspect('equal')
	ax5.set_xlim([-0.5, 0.5])
	ax5.set_ylim([-0.5, 0.5])
	ax5.set_zlim([-0.5, 0.5])

	raw_shape = action_sample[0].shape#np.expand_dims(action_sample[0], axis=0).shape
	hviz.plot(ax5, 
			inverse_mappings[embed_idx].source_data[nearest_frame_num].reshape(raw_shape), 
			lcolor='#099487', 
			rcolor='#F51836')
	# plot_source_poses(a, 0, vis, best_local_idx, np.expand_dims(action_sample[0], axis=0).shape, create_video=create_video, noaxis=noaxis, views=views)
	# ax3.imshow(action_sample[nearest_frame_num])
	fig.canvas.draw()

def hover(event):
	xy = np.array([event.xdata, event.ydata])
	if event.inaxes is ax1:
		embed_idx = 0
	elif event.inaxes is ax2:
		embed_idx = 1
	else:
		return

	display_state(xy, embed_idx)

def keypress(event):
	global color_idx, save_count, nearest_frame_num
	if event.key == "c":
		plt.savefig("saved_%03d.png" % save_count)
		extent = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		plt.savefig('walking_figure_%03d.png' % save_count, bbox_inches=extent)
		save_count += 1
	

fig.canvas.mpl_connect("key_press_event", keypress)
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()