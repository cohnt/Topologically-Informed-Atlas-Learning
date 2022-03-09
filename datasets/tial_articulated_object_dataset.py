import numpy as np
import csv
import cv2
import os

from functools import reduce
from operator import add

from pupil_apriltags import Detector

import src.geometry as geometry

def get_calib(fname, get_distortion=False):
	# Gets calibration data from the given filename, fname. Returns
	# the intrinsic parameters, and optionally the distortion parameters
	# as well. The return is a tuple (fx,fy,cx,cy,d0,d1,d2,d3,d4)
	with open(fname, newline="") as calib_csvfile:
		calib_reader = csv.reader(calib_csvfile, delimiter=",")
		header = next(calib_reader)
		calib_raw = next(calib_reader)
		calib = [float(num) for num in calib_raw]
		if get_distortion:
			return tuple(calib)
		else:
			return tuple(calib[0:4])

def load_video(fname, calib, tag_size, undistort=True, force_reload=False, sharpen_func=None):
	# Reads a video sequence of an object with apriltags, and returns all
	# of the tag poses. This will also save the poses into a csv file, with
	# the same name as fname (but with a .csv extension instead). If this
	# file already exists, it will instead load in the poses from that file,
	# unless force_reload is True, in which case it will delete and recreate
	# that file. This function discards frames where not all tags are detected.

	# fname should be the relative path to the video file, including the extension.
	# calib should be the calibration parameters (fx,fy,cx,cy,d0,d1,d2,d3,d4).
	# If undistort is True, the images will be undistorted using cv2.undistort. If
	# it's false, then calib can just be passed in as (fx,fy,cx,cy).
	# tag_size should be the width of the apriltags used, in mm. See the apriltags
	# user guide for how this can be determined:
	# https://github.com/AprilRobotics/apriltag/wiki/AprilTag-User-Guide#pose-estimation

	# sharpen_func, if specified, should be a function that processes a greyscale image
	# into another greyscale image, and returns the new image.

	# Returns a 2d list of poses, where each "row" is a single frame.

	# Check if the file exists already
	csv_fname = os.path.splitext(fname)[0] + ".csv"
	if os.path.isfile(csv_fname):
		# It exists!
		# Delete it if force_reload is True. Otherwise, just load the poses in.
		if force_reload:
			os.remove(csv_fname)
		else:
			return get_poses_from_csv(csv_fname)

	# Helper variables, to improve readability.
	fx, fy, cx, cy = calib[0:4]
	if undistort:
		dist_coefs = calib[4:]
	camera_mat = np.array([
		[fx, 0, cx],
		[0, fy, cy],
		[0, 0, 1]
	])

	at_detector = Detector(families='tag36h11', nthreads=8, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25, debug=0)

	poses = []

	# Process all of the frames
	vidcap = cv2.VideoCapture(fname)
	success, img = vidcap.read()
	while success:
		# Initial image processing
		if undistort:
			img = cv2.undistort(img, camera_mat, dist_coefs)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		if not sharpen_func is None:
			gray = sharpen_func(gray)

		# Tag detection and pose estimation
		tags = at_detector.detect(gray, estimate_tag_pose=True, camera_params=calib[0:4], tag_size=tag_size)
		idxs = np.argsort([tag.tag_id for tag in tags])
		poses.append([])
		for idx in idxs:
			poses[-1].append(geometry.Pose(tags[idx].pose_t, geometry.mat_to_quat(tags[idx].pose_R)))

		# Get the next image
		success, img = vidcap.read()

	# Sometimes, not every apriltag is detected. We treat the true number of tags
	# as the maximum seen in a single image. Then, we discard any frames with
	# missing tags.
	num_tags = max(*[len(l) for l in poses])

	frame_nums = [i for i in range(len(poses))]

	# Discard frames with missing tags.
	orig_num = len(poses)
	keep_idxs = [i for i in range(len(poses)) if len(poses[i]) == num_tags]
	frame_nums = [frame_nums[i] for i in keep_idxs]
	poses = [poses[i] for i in keep_idxs]
	new_num = len(poses)
	n_discarded = orig_num - new_num
	print("Discarded %d frames" % n_discarded)

	# Save poses to a csv file.
	with open(csv_fname, "w", newline="") as csv_file:
		writer = csv.writer(csv_file, delimiter=",")
		for i in range(len(poses)):
			pose_list = poses[i]
			csv_row = [str(frame_nums[i])] + reduce(add, [pose_to_csv_list(pose) for pose in pose_list])
			writer.writerow(csv_row)

	return poses, frame_nums

def pose_to_csv_list(pose):
	# Converts a geometry.pose object to a list [x,y,z,qx,qy,qz,qw]
	pos = list(geometry.unhomogenize_vectors(pose.pos).flatten())
	quat = list(pose.quat)
	return pos + quat

def csv_list_to_pose(csv_list):
	# Converts a list [x,y,z,qx,qy,qz,qw] to a geometry.pose object
	pos = np.array(csv_list[0:3], dtype=float)
	quat = np.array(csv_list[3:], dtype=float)
	return geometry.Pose(pos, quat)

def get_poses_from_csv(fname):
	# Reads in a set of apriltag poses from the csv file fname.
	# Returns a 2d list of poses, where each "row" is a single frame.

	poses = []
	frame_nums = []
	with open(fname, "r", newline="") as csv_file:
		reader = csv.reader(csv_file, delimiter=",")
		for row in reader:
			poses.append([])
			frame_nums.append(int(row[0]))
			for i in range(1, len(row), 7):
				poses[-1].append(csv_list_to_pose(row[i:i+7]))

	return poses, frame_nums

def pose_to_corners(pose, tag_size):
	# Given a pose, and the size of a tag, returns the 3D coordinates
	# of the tag's corners.
	# Returns 4 3D points, in the form of a (4,3) numpy array
	corners = np.array([
		[1, 1, 0],
		[1, -1, 0],
		[-1, 1, 0],
		[-1, -1, 0]
	], dtype=np.float64).T * tag_size / 2
	corners = np.matmul(geometry.quat_to_mat(pose.quat), corners) + geometry.unhomogenize_vectors(pose.pos)
	return corners.T

def corners_to_pose(corners):
	# Given the positions of the four corners of a tag, return its pose.
	# For the ordering of the corners, see pose_to_corners above.
	# corners should be 4 3D points, in the form of a (4,3) numpy array
	center = np.mean(corners, axis=0)
	x_axis = (np.mean(corners[[0,1],:], axis=0) - center).flatten()
	y_axis = (np.mean(corners[[0,2],:], axis=0) - center).flatten()

	# Ensure the y_axis is orthogonal to the x_axis
	proj = (np.dot(x_axis, y_axis) / np.dot(x_axis, x_axis)) * x_axis
	y_axis = y_axis - proj

	# Normalize the axes
	x_axis = x_axis / np.linalg.norm(x_axis)
	y_axis = y_axis / np.linalg.norm(y_axis)

	# Cross product go brrrrr
	z_axis = np.cross(x_axis, y_axis)
	z_axis = z_axis / np.linalg.norm(z_axis) # Probably unnecessary?

	# Construct the pose object
	rot_mat = np.vstack((x_axis, y_axis, z_axis)).T
	return geometry.Pose(center, geometry.mat_to_quat(rot_mat))

def transform_poses_to_local_frame(poses, reference_idx):
	# For each frame, transforms all poses into the local frame of
	# the pose at index reference_idx
	local_poses = []
	for pose_list in poses:
		local_poses.append([])
		for pose in pose_list:
			pos = geometry.global_xyz_to_local_xyz(pose_list[reference_idx], pose.pos)
			quat = geometry.compose_quat_rotations(geometry.quat_inverse_rotation(pose_list[reference_idx].quat), pose.quat)
			local_poses[-1].append(geometry.Pose(pos, quat))
	return local_poses

def get_video_frame(fname, frame_num):
	# Grabs the given frame number from the video, and returns it as an rgb image.
	vidcap = cv2.VideoCapture(fname)
	vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
	success, img = vidcap.read()
	img = img[:,:,[2,1,0]]
	return img