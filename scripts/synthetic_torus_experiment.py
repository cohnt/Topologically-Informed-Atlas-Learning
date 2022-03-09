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
from sklearn.manifold import Isomap, trustworthiness

import src.visualization as visualization
import src.atlas as atlas
import src.cyclecut as cyclecut
import src.util as util
import datasets.synthetic_manifolds as synthetic_manifolds

seed = 43
np.random.seed(seed)

dataset_func = synthetic_manifolds.make_torus # Change this
n_points = 4000
neighbors_k = 12
noise = 0

data, param = dataset_func(n_points, noise)
# idx1 = data[:,0] > -0.1
# idx2 = data[:,0] < 0.1
# idx = np.logical_and(idx1, idx2)
# data = data[idx]
# param = param[idx]
color_param = param[:,1] # Change this
dim = data.shape[1]
adj_mat = kneighbors_graph(data, neighbors_k, mode="distance").toarray()
adj_mat = np.maximum(adj_mat, adj_mat.T)

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection="3d")
# ax.scatter(data[:,0], data[:,1], data[:,2])
# visualization.draw_neighbors(ax, data, adj_mat)
# plt.show()

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
ax.view_init(elev=75)
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

charts = a.get_all_chart_domains()
a.construct_inverse_mapping()

fig = plt.figure()
n_cols = len(charts) + 1
tl_ax = fig.add_subplot(2, n_cols, 1, projection="3d")
top_axes = [fig.add_subplot(2, n_cols, i, projection="3d") for i in range(2, n_cols+1)]
bl_ax = fig.add_subplot(2, n_cols, n_cols+1)
bottom_axes = [fig.add_subplot(2, n_cols, n_cols+i) for i in range(2, n_cols+1)]

util.maximize_plt_fig(fig)

# Draw edges
# visualization.draw_neighbors(ax1, data, adj_mat, c="grey", linewidth=lw)
# visualization.draw_neighbors(ax2, charts[0][0], charts[0][1], c="grey", linewidth=lw)
# visualization.draw_neighbors(ax3, charts[1][0], charts[1][1], c="grey", linewidth=lw)

cmap = plt.get_cmap("hsv")
param_idx = 0
msl = 0.5

# Draw points
tl_ax.scatter(data[:,0], data[:,1], data[:,2], c=param[:,param_idx], cmap=cmap, s=pr**2)
ism = Isomap(n_neighbors=neighbors_k, n_components=2)
raw_embedding = ism.fit_transform(data)
print("Isomap error: %s" % ism.reconstruction_error())
print("Isomap trustworthiness: %s" % trustworthiness(data, raw_embedding))
bl_ax.scatter(raw_embedding[:,0], raw_embedding[:,1], c=param[:,param_idx], cmap=cmap, vmin=0, vmax=1)

for i in range(len(charts)):
	ax = top_axes[i]
	ax.scatter(charts[i][0][:,0], charts[i][0][:,1], charts[i][0][:,2], s=pr**2)

for i in range(len(charts)):
	ax = bottom_axes[i]
	ax.scatter(a.embeddings[i][:,0], a.embeddings[i][:,1], c=param[np.array(list(a.atlas[a.atlas_idx[i]])),param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)
	# visualization.draw_triangles(ax, a.embeddings[i], a.inverses[i].tri, param[np.array(list(a.atlas[a.atlas_idx[i]])),param_idx], cmap=cmap, edge_color="black", lw=0.5, max_side_len=msl)

# def on_move(event):
# 	if event.inaxes == ax1:
# 		ax2.view_init(elev=ax1.elev, azim=ax1.azim)
# 		ax3.view_init(elev=ax1.elev, azim=ax1.azim)
# 		ax4.view_init(elev=ax1.elev, azim=ax1.azim)
# 	elif event.inaxes == ax2:
# 		ax1.view_init(elev=ax2.elev, azim=ax2.azim)
# 		ax3.view_init(elev=ax2.elev, azim=ax2.azim)
# 		ax4.view_init(elev=ax2.elev, azim=ax2.azim)
# 	elif event.inaxes == ax3:
# 		ax1.view_init(elev=ax3.elev, azim=ax3.azim)
# 		ax2.view_init(elev=ax3.elev, azim=ax3.azim)
# 		ax4.view_init(elev=ax3.elev, azim=ax3.azim)
# 	elif event.inaxes == ax4:
# 		ax1.view_init(elev=ax4.elev, azim=ax4.azim)
# 		ax2.view_init(elev=ax4.elev, azim=ax4.azim)
# 		ax3.view_init(elev=ax4.elev, azim=ax4.azim)
# 	else:
# 		return
# 	fig.canvas.draw_idle()
# c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.draw()
plt.pause(1)
plt.savefig("output.png")
plt.show()