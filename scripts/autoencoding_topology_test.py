#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

# Some of this code is heavily based on https://github.com/ekorman/Atlas/blob/master/Atlas_demo.ipynb

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import Isomap

import src.visualization as visualization
import src.util as util
import src.comparisons.autoencoder_atlas as atlas
import datasets.synthetic_manifolds as synthetic_manifolds

from src.cyclecut import find_large_atomic_cycle

seed = 43
np.random.seed(seed)

dataset_func = synthetic_manifolds.make_torus # Change this
n_points = 4000
neighbors_k = 12
noise = 0

# Initial dataset stuff

data, param = dataset_func(n_points, noise)
color_param = param[:,1] # Change this
dim = data.shape[1]

cmap = plt.get_cmap("hsv")

# Learn an atlas

target_dim = 2
n_charts = 4

at = atlas.Atlas(target_dim, n_charts)
at.fit(data)

# Plot chart assignments

probs = at.chart_probs(data)
chart_assignments = probs.argmax(axis=1)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
util.maximize_plt_fig(fig)
ax.scatter(data[:,0], data[:,1], data[:,2], c=chart_assignments)
plt.show()

for i in range(n_charts):
	if len(data[chart_assignments==i,0]) == 0:
		continue
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1, projection='3d')
	util.maximize_plt_fig(fig)
	ax.scatter(data[chart_assignments==i,0], data[chart_assignments==i,1], data[chart_assignments==i,2], c=param[chart_assignments==i,1], cmap=cmap, s=10**2)
	plt.show()

# Plot chart embeddings

from keras import backend as K

embs = at.encoder(K.constant(data))
np_embs = [K.eval(emb) for emb in embs]

chart_embs = [np_embs[i][chart_assignments == i,:] for i in range(n_charts)]

from sklearn.manifold import trustworthiness
trusts = []
for i in range(len(chart_embs)):
	emb = chart_embs[i]
	if len(emb) == 0:
		continue
	orig = data[chart_assignments==i,:]
	trusts.append(trustworthiness(orig, emb))

print("Chart trustworthinesses: %s" % trusts)
print("Mean trustworthiness: %s" % np.mean(trusts))
print("Min trustworthiness: %s" % np.min(trusts))

for i in range(n_charts):
	chart_emb = chart_embs[i]
	if len(chart_emb) == 0:
		continue
	fig = plt.figure()
	ax1 = fig.add_subplot(2, 2, 1)
	ax2 = fig.add_subplot(2, 2, 2)
	ax3 = fig.add_subplot(2, 2, 3)
	ax4 = fig.add_subplot(2, 2, 4)
	util.maximize_plt_fig(fig)
	ax1.scatter(chart_emb[:,0], chart_emb[:,1], c=param[chart_assignments == i,0], cmap=cmap)
	ax2.scatter(chart_emb[:,0], chart_emb[:,1], c=param[chart_assignments == i,1], cmap=cmap)
	plt.show()

# Plot chart reconstructions

# gendata = np.arange(-1,1,.05)
# sq = np.array([[x,y] for x in gendata for y in gendata])

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection='3d')
# util.maximize_plt_fig(fig)
# for i in range(n_charts):
# 	preds = at.decode(i, sq)
# 	ax.scatter(preds[:,0], preds[:,1], preds[:,2], alpha=0.7)
# plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
util.maximize_plt_fig(fig)
for i in range(n_charts):
	if len(chart_embs[i]) == 0:
		continue
	preds = at.decode(i, chart_embs[i])
	ax1.scatter(preds[:,0], preds[:,1], preds[:,2], alpha=0.7, c=param[chart_assignments == i,0], cmap=cmap)
	ax2.scatter(preds[:,0], preds[:,1], preds[:,2], alpha=0.7, c=param[chart_assignments == i,1], cmap=cmap)
plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(2,5,1,projection="3d")
# ax2 = fig.add_subplot(2,5,6)
# util.maximize_plt_fig(fig)
# ax1.scatter(data[chart_assignments==1,0], data[chart_assignments==1,1], data[chart_assignments==1,2], c=param[chart_assignments==1,1], cmap=cmap, s=5**2)
# ax2.scatter(chart_embs[1][:,0], chart_embs[1][:,1], c=param[chart_assignments==1,1], cmap=cmap)
# plt.show()