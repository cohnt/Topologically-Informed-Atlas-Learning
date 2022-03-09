#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import Isomap

import src.visualization as visualization
import src.atlas as atlas
import src.cyclecut as cyclecut
import src.util as util
import datasets.synthetic_manifolds as synthetic_manifolds

seed = 43
np.random.seed(seed)

dataset_func = synthetic_manifolds.make_swiss_roll # Change this
n_points = 3000
neighbors_k = 12
noise = 0

data, param = dataset_func(n_points, noise)
color_param = param[:,1] # Change this
dim = data.shape[1]
adj_mat = kneighbors_graph(data, neighbors_k, mode="distance").toarray()
adj_mat = np.maximum(adj_mat, adj_mat.T)

# Display parameters
lw = 1.0
pr = 5.0

# Pointwise chart seeding

L = 8
def hole_detector(adj_mat):
	return len(cyclecut.find_large_atomic_cycle(adj_mat, L)) > 0

a = atlas.Atlas(data, adj_mat, 2, hole_detector, Isomap(n_components=2, metric="precomputed", n_neighbors=neighbors_k))
# a.seed_charts_random(25, 2)
# a.seed_charts_pointwise(2)
a.seed_charts_iterative_farthest_point(25, 2)

# fig = plt.figure()
# ax = fig.add_subplot(projection="3d")
# ax.view_init(elev=60)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
ax.view_init(elev=10, azim=90)
util.maximize_plt_fig(fig)
while len(a.atlas) > 1:
	ax.cla()
	visualization.draw_non_overlap_domains(ax, a.data, a.non_overlap_charts, a.original_chart_count, s=10**2)
	plt.draw()
	plt.pause(0.01)
	plt.savefig("frame_%03d.png" % (a.original_chart_count - len(a.atlas)))

	# if not a.combine_charts_minimum(both=True):
	# 	if not a.combine_charts_minimum(both=False):
	# 		if not a.combine_charts_exhaustive():
	# 			break
	if not a.combine_charts_exhaustive():
		break
	print(len(a.atlas))
	# charts = a.get_all_chart_domains()
	# chart_labels = a.atlas_idx
	# visualization.draw_atlas_domains(ax, charts, chart_labels)
	# ax.scatter(charts[0][0][:,0], charts[0][0][:,1], charts[0][0][:,2], c="red", s=pr**2)
	# ax.scatter(charts[1][0][:,0], charts[1][0][:,1], charts[1][0][:,2], c="blue", s=pr**2)
	# plt.draw()
	# plt.pause(0.001)

if len(a.atlas) == 1:
	ax.cla()
	visualization.draw_non_overlap_domains(ax, a.data, a.non_overlap_charts, a.original_chart_count, s=10**2)
	plt.draw()
	plt.pause(1)
	# plt.savefig("frame_%03d.png" % (a.original_chart_count - len(a.atlas)))

plt.show()

a.embed_charts()
print("Chart errors: %s" % a.chart_errors)
print("Chart trustworthinesses: %s" % a.chart_trustworthinesses)
print("Mean trustworthiness: %s" % np.mean(a.chart_trustworthinesses))
print("Min trustworthiness: %s" % np.min(a.chart_trustworthinesses))

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.view_init(elev=10, azim=90)
ax2 = fig.add_subplot(1, 2, 2)
util.maximize_plt_fig(fig)

charts = a.get_all_chart_domains()
a.construct_inverse_mapping()

cmap = plt.get_cmap("rainbow")
param_idx = 1
msl = 1

# Draw points
ax1.scatter(data[:,0], data[:,1], data[:,2], c=param[:,param_idx], cmap=cmap, s=pr**2)
ax2.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=param[np.array(list(a.atlas[a.atlas_idx[0]])),param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)
# visualization.draw_triangles(ax2, a.embeddings[0], a.inverses[0].tri, param[np.array(list(a.atlas[a.atlas_idx[0]])),param_idx], cmap=cmap, edge_color="black", lw=0.5, max_side_len=msl)

plt.draw()
plt.pause(1)
plt.savefig("output.png")
plt.show()