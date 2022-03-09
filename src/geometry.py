import numpy as np
import scipy, scipy.spatial
import cv2

class Pose():
	def __init__(self, pos, quat):
		pos = np.array(pos).reshape(-1,1)
		if len(pos) == 3:
			pos = homogenize_vectors(pos)
		self.pos = pos # Should be a homogeneous 4x1 column vector
		self.quat = np.array(quat) / np.linalg.norm(quat) # Normalize to a unit quaternion.
		                                        # Format is [x,y,z,w] for the quaternion xi+yj+zk+w.
		                                        # This "scalar-last" convention is used by scipy.

	def __repr__(self):
		return "<Pose pos:%s quat:%s>" % (self.pos.flatten()[:-1], self.quat)

def identity_pose():
	return Pose([0,0,0], identity_quat())

def identity_quat():
	return np.array([0,0,0,1])

def quat_error(q1, q2):
	return min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 + q2))

def quat_to_mat(quat):
	# Converts a quaternion w+xi+yj+zk stored as [x,y,z,w] to a 3x3 rotation matrix
	return scipy.spatial.transform.Rotation.from_quat(quat).as_matrix()

def mat_to_quat(mat):
	# Converts a 3x3 rotation matrix to a quaternion w+xi+yj+zk stored as [x,y,z,w]
	return scipy.spatial.transform.Rotation.from_matrix(mat).as_quat()

def rot_vec_to_mat(rot_vec):
	# Converts a 3x1 rotation vector to a 3x3 rotation matrix
	mat, _ = cv2.Rodrigues(rot_vec)
	return mat

def rot_vec_to_quat(rot_vec):
	# Converts a 3x1 rotation vector to a quaternion w+xi+yj+zk stored as [x,y,z,w]
	mat, _ = cv2.Rodrigues(rot_vec)
	return mat_to_quat(mat)

def mat_to_rot_vec(mat):
	# Converts a 3x3 rotation matrix to a 3x1 rotation vector
	rot_vec, _ = cv2.Rodrigues(mat)
	return rot_vec

def quat_to_rot_vec(quat):
	# Converts a quaternion w+xi+yj+zk stored as [x,y,z,w] to a 3x3 rotation matrix
	return mat_to_rot_vec(quat_to_mat(quat))

def homogenize_matrix(mat):
	# Converts a 3x3 matrix to a 4x4 homogeneous matrix
	new_mat = np.zeros((4,4))
	new_mat[0:3,0:3] = mat
	new_mat[3,3] = 1
	return new_mat

def unhomogenize_matrix(mat):
	# Converts a 4x4 homogeneous matrix to a 3x3 matrix
	return mat[0:3,0:3]

def homogenize_vectors(vecs):
	# Converts a 3xn set of vectors to a 4xn set of homogeneous vectors
	return np.pad(vecs, pad_width=((0,1),(0,0)), mode="constant", constant_values=1)

def unhomogenize_vectors(vecs):
	# Converts a 4xn set of homogeneous vectors to a 3xn set of vectors
	return vecs[0:3,:]

def homogeneous_norm(vec):
	# Returns the l2-norm of a 4x1 or 3x1 homogeneous vector
	return np.linalg.norm(vec.flatten()[:-1])

def make_translation_matrix(vec):
	# Converts a 3x1 vector or 4x1 homogeneous vector to a 4x4 homogeneous translation matrix
	mat = np.eye(4)
	mat[0:3,3] = vec.flatten()[0:3]
	return mat

def local_xyz_to_global_xyz(local_pose, xyz):
	# Transforms 3d points from the global frame to the local frame
	# xyz should be a 4x1 set of homogeneous vectors in the global frame
	# Will return a 4x1 set of homogeneous vectors in the local frame
	rot_mat = homogenize_matrix(quat_to_mat(local_pose.quat))
	trans_mat = make_translation_matrix(local_pose.pos)
	full_mat = np.matmul(trans_mat, rot_mat) # Rotate, then translate
	return np.matmul(full_mat, xyz)

def global_xyz_to_local_xyz(local_pose, xyz):
	# Transforms 3d points from the local frame to the global frame
	# xyz should be a 4x1 set of homogeneous vectors in the local frame
	# Will return a 4x1 set of homogeneous vectors in the global frame
	rot_mat = homogenize_matrix(quat_to_mat(local_pose.quat)).T
	trans_mat = make_translation_matrix(-1 * local_pose.pos)
	full_mat = np.matmul(rot_mat, trans_mat)
	return np.matmul(full_mat, xyz)

def quat_inverse_rotation(q):
	# Returns the quaternion corresponding to the opposite rotation.
	return mat_to_quat(quat_to_mat(q).T)

def compose_quat_rotations(q1, q2):
	# Converts two quaternions to rotation matrices, multiplies them,
	# and then converts it back into a quaternion.
	m1 = quat_to_mat(q1)
	m2 = quat_to_mat(q2)
	m_out = np.matmul(m1, m2)
	return mat_to_quat(m_out)