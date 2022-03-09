#######################
# Fix import issues   #
import sys            #
sys.path.append(".")  #
sys.path.append("..") #
#######################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from sklearn.utils.graph_shortest_path import graph_shortest_path
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap, trustworthiness

import src.visualization as visualization
import src.atlas as atlas
import src.cyclecut as cyclecut
import src.geometry as geometry
import src.util as util

import datasets.tial_articulated_object_dataset as tial

seed = 43
np.random.seed(seed)

tag_size = 0.52 * 25.4 # Tags are printed in inches, so have to convert to mm
reference_frame_id = 0 # Index of the tag used as the local reference frame
hinge_dist_tag_id = 1 # The tag used for the hinge distance
rot_dist_tag_id = 5 # The tag used for the rotation distance

neighbors_mode = "knn" # Choices: "knn" or "radius"
neighbors_k = 35
neighbors_rad = 15
target_dim = 2

atlas_L = 8
n_initial_charts = 15
atlas_overlap = 1

downsample = 2
start_idx = 0
end_idx = -1

calib_fname = "./datasets/articulated_objects/tial_articulated_object_dataset/calib_tommy_phone.csv"
video_fname = "./datasets/articulated_objects/tial_articulated_object_dataset/corkscrew.mp4"

def hole_detector(adj_mat):
	print("Hole detector running...", end="")
	cycle = cyclecut.find_large_atomic_cycle(adj_mat, atlas_L)
	print("Done!")
	return len(cycle) > 0

# Get the apriltag pose data
print("Loading data...")
calib = tial.get_calib(calib_fname, get_distortion=True)
poses, frame_nums = tial.load_video(video_fname, calib, tag_size, sharpen_func=util.strong_sharpen_func)
poses = poses[start_idx:end_idx:downsample]
frame_nums = frame_nums[start_idx:end_idx:downsample]
poses = tial.transform_poses_to_local_frame(poses, reference_frame_id)

# Get the two useful colorings to interpret the data
hinge_dist = np.array([np.linalg.norm(pose_list[hinge_dist_tag_id].pos) for pose_list in poses])
x_axis = np.array([geometry.quat_to_mat(pose_list[rot_dist_tag_id].quat)[:,1] for pose_list in poses])
rot_dist = np.arctan2(x_axis[:,2], x_axis[:,1])

# Extract the tag center locations to use with atlas learning
data = []
for pose_list in poses:
	data.append([])
	for pose in pose_list:
		data[-1] = data[-1] + list(tial.pose_to_corners(pose, tag_size).flatten())
data = np.array(data)

print("Using %d points." % len(data))

# Construct nearest neighbors
print("Constructing neighborhood graph...")
if neighbors_mode == "knn":
	adj_mat = kneighbors_graph(data, n_neighbors=neighbors_k, mode="distance", n_jobs=-1).toarray()
elif neighbors_mode == "radius":
	adj_mat = radius_neighbors_graph(data, radius=neighbors_rad, mode="distance", n_jobs=-1).toarray()
adj_mat = np.maximum(adj_mat, adj_mat.T)

print("Using %d edges." % (np.sum(np.nonzero(adj_mat)) / 2))

viz_points = np.vstack((hinge_dist, rot_dist)).T
viz_points_3d = np.vstack((hinge_dist, np.cos(rot_dist), np.sin(rot_dist))).T

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.scatter(viz_points[:,0], viz_points[:,1])
# ax.set_xlabel("Corkscrew Handle Angle")
# ax.set_ylabel("Corkscrew Twist Angle")
# visualization.draw_neighbors(ax, viz_points, adj_mat, c="blue")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection="3d")
# ax.scatter(viz_points_3d[:,0], viz_points_3d[:,1], viz_points_3d[:,2])
# visualization.draw_neighbors(ax, viz_points_3d, adj_mat, c="blue")
# plt.show()

print("Initializing atlas...")
a = atlas.Atlas(data, adj_mat, 2, hole_detector, Isomap(n_components=2, metric="precomputed"), viz_points_3d)
a.seed_charts_iterative_farthest_point(n_initial_charts, atlas_overlap, metric=False)
# a.seed_charts_random(n_initial_charts, atlas_overlap)

print("Combining charts...")
while len(a.atlas) > 1:
	if not a.combine_charts_exhaustive():
		break
	print(len(a.atlas))

print("Embedding charts...")
a.embed_charts_kpca()
print("Chart errors: %s" % a.chart_errors)
print("Chart trustworthinesses: %s" % a.chart_trustworthinesses)
print("Mean trustworthiness: %s" % np.mean(a.chart_trustworthinesses))
print("Min trustworthiness: %s" % np.min(a.chart_trustworthinesses))
ism = Isomap(n_neighbors=neighbors_k, n_components=2)
raw_embedding = ism.fit_transform(data)
print("Isomap error: %s" % ism.reconstruction_error())
print("Isomap trustworthiness: %s" % trustworthiness(data, raw_embedding))

# Autoencoder comparison
# import src.comparisons.autoencoder_atlas as atlas
# target_dim = 2
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
# 	if len(emb) < 6:
# 		continue
# 	orig = data[chart_assignments==i,:]
# 	trusts.append(trustworthiness(orig, emb))
# print("Chart trustworthinesses: %s" % trusts)
# print("Mean trustworthiness: %s" % np.mean(trusts))
# print("Min trustworthiness: %s" % np.min(trusts))

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# for chart in a.atlas.items():
# 	idxs = np.array(list(chart[1]))
# 	points = viz_points[idxs]
# 	ax.scatter(points[:,0], points[:,1])
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
for chart in a.atlas.items():
	idxs = np.array(list(chart[1]))
	points = viz_points_3d[idxs]
	ax.scatter(points[:,0], points[:,1], points[:,2])
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

hinge_dist_norm = (hinge_dist - np.min(hinge_dist)) / (np.max(hinge_dist) - np.min(hinge_dist))
rot_dist_norm = (rot_dist - np.min(rot_dist)) / (np.max(rot_dist) - np.min(rot_dist))

hinge_vmin = np.min(hinge_dist_norm)
hinge_vmax = np.max(hinge_dist_norm)
rot_vmin = np.min(rot_dist_norm)
rot_vmax = np.max(rot_dist_norm)

# for idx in a.atlas_idx:
# 	print(np.min(rot_dist_norm[np.array(list(a.atlas[idx]))]))
# 	print(np.max(rot_dist_norm[np.array(list(a.atlas[idx]))]))
# 	print("")

ax1.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=hinge_dist_norm[np.array(list(a.atlas[a.atlas_idx[0]]))], cmap=plt.get_cmap("rainbow"), vmin=hinge_vmin, vmax=hinge_vmax)
ax2.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=rot_dist_norm[np.array(list(a.atlas[a.atlas_idx[0]]))], cmap=plt.get_cmap("hsv"), vmin=rot_vmin, vmax=rot_vmax)
ax1.set_title("Chart 1, Non-Loopy Component")
ax2.set_title("Chart 1, Loopy Component")

if len(a.embeddings) > 1:
	ax3.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=hinge_dist_norm[np.array(list(a.atlas[a.atlas_idx[1]]))], cmap=plt.get_cmap("rainbow"), vmin=hinge_vmin, vmax=hinge_vmax)
	ax4.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=rot_dist_norm[np.array(list(a.atlas[a.atlas_idx[1]]))], cmap=plt.get_cmap("hsv"), vmin=rot_vmin, vmax=rot_vmax)
	ax3.set_title("Chart 2, Non-Loopy Component")
	ax4.set_title("Chart 2, Loopy Component")

plt.show()

# Now do the inverse mapping stuff...
a.construct_inverse_mapping()
fig = plt.figure()
util.maximize_plt_fig(fig)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3, projection="3d")

all_points = data.reshape(len(data),-1,3)
centers = 0.5 * np.array([np.max(all_points[:,:,i]) + np.min(all_points[:,:,i]) for i in range(3)])
ranges = np.array([np.max(all_points[:,:,i]) - np.min(all_points[:,:,i]) for i in range(3)])
max_range = np.max(ranges)
xlims = [centers[0] - (max_range/2), centers[0] + (max_range/2)]
ylims = [centers[1] - (max_range/2), centers[1] + (max_range/2)]
zlims = [centers[2] - (max_range/2), centers[2] + (max_range/2)]

params = np.vstack((hinge_dist_norm, rot_dist_norm)).T
cmaps = [plt.get_cmap("rainbow"), plt.get_cmap("hsv")]
color_idx = 0
save_count = 0

nearest_frame_num = 0

ax1.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=params[np.array(list(a.atlas[a.atlas_idx[0]])),color_idx], cmap=cmaps[color_idx], vmin=hinge_vmin, vmax=hinge_vmax)
ax2.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=params[np.array(list(a.atlas[a.atlas_idx[1]])),color_idx], cmap=cmaps[color_idx], vmin=rot_vmin, vmax=rot_vmax)

def keypress(event):
	global color_idx, save_count, nearest_frame_num
	if event.key == " ":
		color_idx = 1 - color_idx
		ax1.cla()
		ax2.cla()
		ax1.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=params[np.array(list(a.atlas[a.atlas_idx[0]])),color_idx], cmap=cmaps[color_idx], vmin=hinge_vmin, vmax=hinge_vmax)
		ax2.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=params[np.array(list(a.atlas[a.atlas_idx[1]])),color_idx], cmap=cmaps[color_idx], vmin=rot_vmin, vmax=rot_vmax)
		fig.canvas.draw()
	elif event.key == "c":
		plt.savefig("saved_%03d.png" % save_count)
		plt.imsave("real_%03d.png" % save_count, tial.get_video_frame(video_fname, nearest_frame_num))
		save_count += 1

def display_state(xy, embed_idx):
	global color_idx, nearest_frame_num

	try:
		orig_point = a.inverses[embed_idx].single_inverse_mapping(xy)
		transition = a.transition_map(embed_idx, xy)
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
	ax1.scatter(a.embeddings[0][:,0], a.embeddings[0][:,1], c=params[np.array(list(a.atlas[a.atlas_idx[0]])),color_idx], cmap=cmaps[color_idx], vmin=hinge_vmin, vmax=hinge_vmax)
	ax2.scatter(a.embeddings[1][:,0], a.embeddings[1][:,1], c=params[np.array(list(a.atlas[a.atlas_idx[1]])),color_idx], cmap=cmaps[color_idx], vmin=rot_vmin, vmax=rot_vmax)
	if len(transition[0]) > 0:
		ax1.scatter([transition[0][0]], [transition[0][1]], color="black", marker="+", s=25**2)
	if len(transition[1]) > 0:
		ax2.scatter([transition[1][0]], [transition[1][1]], color="black", marker="+", s=25**2)

	all_corners = orig_point.reshape(-1,4,3)
	poses = [tial.corners_to_pose(corners) for corners in all_corners]
	points = np.array([geometry.unhomogenize_vectors(pose.pos) for pose in poses]).reshape(-1,3)

	ax3.cla()
	ax3.set_xlim(xlims)
	ax3.set_ylim(ylims)
	ax3.set_zlim(zlims)
	ax3.plot(points[[1,2],0], points[[1,2],1], points[[1,2],2], c="black") # Show the connected parts
	ax3.plot(points[[3,4],0], points[[3,4],1], points[[3,4],2], c="black") # Show the connected parts
	ax3.plot(points[[5,6],0], points[[5,6],1], points[[5,6],2], c="black") # Show the connected parts
	ax3.scatter(orig_point[0::3], orig_point[1::3], orig_point[2::3], c="orange") # Show the corners
	# ax3.scatter(points[:,0], points[:,1]) # Show the centers
	# Draw the coordinate frames
	axes_len = 8
	for i in range(len(poses)):
		pose = poses[i]
		x = np.array([[0, 0, 0], [1, 0, 0]]).T * axes_len
		y = np.array([[0, 0, 0], [0, -1, 0]]).T * axes_len
		z = np.array([[0, 0, 0], [0, 0, -1]]).T * axes_len

		# Transform to camera frame
		R = geometry.quat_to_mat(pose.quat)
		t = geometry.unhomogenize_vectors(pose.pos)
		x = np.matmul(R, x) + t
		y = np.matmul(R, y) + t
		z = np.matmul(R, z) + t

		ax3.plot(x[0,:], x[1,:], x[2,:], c="red")
		ax3.plot(y[0,:], y[1,:], y[2,:], c="green")
		ax3.plot(z[0,:], z[1,:], z[2,:], c="blue")

		ax3.text(t[0,0]+20, t[1,0], t[2,0], str(i), fontsize="medium")

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

fig.canvas.mpl_connect("key_press_event", keypress)
fig.canvas.mpl_connect("motion_notify_event", hover)
plt.show()