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

dataset_func = synthetic_manifolds.make_cylinder # Change this
n_points = 1000
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

fig = plt.figure()
ax1 = fig.add_subplot(2, 3, 1, projection="3d")
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
ax3 = fig.add_subplot(2, 3, 3, projection="3d")
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

xlim = [np.min(data[:,0]), np.max(data[:,0])]
ylim = [np.min(data[:,1]), np.max(data[:,1])]
zlim = [np.min(data[:,2]), np.max(data[:,2])]

ax1.view_init(elev=75)
ax2.view_init(elev=75)
ax3.view_init(elev=75)

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_zlim(zlim)
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_zlim(zlim)
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_zlim(zlim)

util.maximize_plt_fig(fig)

charts = a.get_all_chart_domains()
a.construct_inverse_mapping()

# Draw edges
# visualization.draw_neighbors(ax1, data, adj_mat, c="grey", linewidth=lw)
# visualization.draw_neighbors(ax2, charts[0][0], charts[0][1], c="grey", linewidth=lw)
# visualization.draw_neighbors(ax3, charts[1][0], charts[1][1], c="grey", linewidth=lw)

cmap = plt.get_cmap("hsv")
param_idx = 1
msl = 0.5

# Draw points
ax1.scatter(data[:,0], data[:,1], data[:,2], c=param[:,param_idx], cmap=cmap, s=pr**2)
ax2.scatter(charts[0][0][:,0], charts[0][0][:,1], charts[0][0][:,2], c="red", s=pr**2)
# visualization.draw_triangles(ax2, charts[0][0], a.inverses[0].tri, param[np.array(list(a.atlas[a.atlas_idx[0]])),param_idx], cmap=cmap, edge_color="black", lw=0.5, max_side_len=msl)
if len(charts) > 1:
	ax3.scatter(charts[1][0][:,0], charts[1][0][:,1], charts[1][0][:,2], c="blue", s=pr**2)
	# visualization.draw_triangles(ax3, charts[1][0], a.inverses[1].tri, param[np.array(list(a.atlas[a.atlas_idx[1]])),param_idx], cmap=cmap, edge_color="black", lw=0.5, max_side_len=msl)

ism = Isomap(n_neighbors=neighbors_k, n_components=2)
raw_embedding = ism.fit_transform(data)
print("Isomap error: %s" % ism.reconstruction_error())
print("Isomap trustworthiness: %s" % trustworthiness(data, raw_embedding))

ax4.scatter(raw_embedding[:,0], raw_embedding[:,1], c=param[:,param_idx], cmap=cmap, vmin=0, vmax=1)
ax5.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=param[np.array(list(a.atlas[a.atlas_idx[0]])),param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)
# visualization.draw_triangles(ax5, a.embeddings[0], a.inverses[0].tri, param[np.array(list(a.atlas[a.atlas_idx[0]])),param_idx], cmap=cmap, edge_color="black", lw=0.5, max_side_len=msl)
if len(charts) > 1:
	ax6.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=param[np.array(list(a.atlas[a.atlas_idx[1]])),param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)
	# visualization.draw_triangles(ax6, a.embeddings[1], a.inverses[1].tri, param[np.array(list(a.atlas[a.atlas_idx[1]])),param_idx], cmap=cmap, edge_color="black", lw=0.5, max_side_len=msl)

def on_move(event):
	if event.inaxes == ax1:
		ax2.view_init(elev=ax1.elev, azim=ax1.azim)
		ax3.view_init(elev=ax1.elev, azim=ax1.azim)
	elif event.inaxes == ax2:
		ax1.view_init(elev=ax2.elev, azim=ax2.azim)
		ax3.view_init(elev=ax2.elev, azim=ax2.azim)
	elif event.inaxes == ax3:
		ax1.view_init(elev=ax3.elev, azim=ax3.azim)
		ax2.view_init(elev=ax3.elev, azim=ax3.azim)
	else:
		return
	fig.canvas.draw_idle()
c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.draw()
plt.pause(1)
plt.savefig("output.png")
plt.show()