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

dataset_func = synthetic_manifolds.make_SO3 # Change this
n_points = 2000
neighbors_k = 12
noise = 0

data, param = dataset_func(n_points, noise)
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

a = atlas.Atlas(data, adj_mat, 3, hole_detector, Isomap(n_components=3, metric="precomputed", n_neighbors=neighbors_k))
# a.seed_charts_random(25, 2)
# a.seed_charts_pointwise(2)
a.seed_charts_iterative_farthest_point(50, 2)

while len(a.atlas) > 1:
	# if not a.combine_charts_minimum(both=True):
	# 	if not a.combine_charts_minimum(both=False):
	# 		if not a.combine_charts_exhaustive():
	# 			break
	if not a.combine_charts_exhaustive():
		break
	print(len(a.atlas))

a.embed_charts()
print("Chart errors: %s" % a.chart_errors)
print("Chart trustworthinesses: %s" % a.chart_trustworthinesses)
print("Mean trustworthiness: %s" % np.mean(a.chart_trustworthinesses))
print("Min trustworthiness: %s" % np.min(a.chart_trustworthinesses))

charts = a.get_all_chart_domains()
a.construct_inverse_mapping()

ism = Isomap(n_neighbors=neighbors_k, n_components=3)
raw_embedding = ism.fit_transform(data)
print("Isomap error: %s" % ism.reconstruction_error())
print("Isomap trustworthiness: %s" % trustworthiness(data, raw_embedding))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
util.maximize_plt_fig(fig)

cmap = plt.get_cmap("hsv")
param_idx = 0
chart_idx = 0
draw_ism = False
msl = 0.5

# Draw points

ax.scatter(a.embeddings[chart_idx][:,0], a.embeddings[chart_idx][:,1], a.embeddings[chart_idx][:,2], c=param[np.array(list(a.atlas[a.atlas_idx[chart_idx]])),param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)

def keypress(event):
	global param_idx, chart_idx, draw_ism
	if event.key == " ":
		param_idx = (param_idx + 1) % 3
	elif event.key == "i":
		draw_ism = True
	elif event.key == "c":
		draw_ism = False
	elif event.key == "n":
		draw_ism = False
		chart_idx = (chart_idx + 1) % len(charts)

	ax.cla()
	if draw_ism:
		ax.scatter(raw_embedding[:,0], raw_embedding[:,1], raw_embedding[:,2], c=param[:,param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)
	else:
		ax.scatter(a.embeddings[chart_idx][:,0], a.embeddings[chart_idx][:,1], a.embeddings[chart_idx][:,2], c=param[np.array(list(a.atlas[a.atlas_idx[chart_idx]])),param_idx], cmap=cmap, zorder=10000, vmin=0, vmax=1, lw=0.5)
	fig.canvas.draw()

fig.canvas.mpl_connect("key_press_event", keypress)

plt.show()