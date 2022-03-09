import numpy as np

def normalize_param_range(param, param_range):
	# Normalizes the parameters into a (0,1) range
	return (param - param_range[0]) / (param_range[1] - param_range[0])

def create_noise(n_samples, dim, noise_factor):
	noise_mean = np.zeros(dim)
	noise_cov = np.eye(dim) * noise_factor
	return np.random.multivariate_normal(noise_mean, noise_cov, n_samples)

# EVERY DATASET FUNCTION HAS THE SAME ARGUMENTS:
# n_sample is the number of points to generate.
# noise_factor is the variance of the Gaussian noise added to each point.
# random_seed is the seed used for the RNG -- use None unless you have a
# particular seed you care about.
# Some dataset functions will have additional, optional arguments.

def make_s_curve(n_samples, noise_factor):
	# The s curve is parameterized by
	# x(t) = 0.5 + sin(t)cos(t)
	# y(t) = 0.5 + 0.5cos(t)
	# for (3/4)pi <= t <= (9/4)pi

	t_range = (3.0 * np.pi / 4.0,
	           9.0 * np.pi / 4.0)
	t = np.random.uniform(*t_range, n_samples)
	xy = np.array([
		0.5 + np.multiply(np.sin(t), np.cos(t)),
		0.5 + (0.5 * np.cos(t))
	]).transpose()

	noise = create_noise(n_samples, 2, noise_factor)
	normalized_t = normalize_param_range(t, t_range)
	return (xy + noise, normalized_t)

def make_circle(n_samples, noise_factor):
	# The circle is parameterized by
	# x(t) = cos(t)
	# y(t) = sin(t)
	# for 0 <= t <= 2pi

	t_range = (0, 2*np.pi)
	t = np.random.uniform(*t_range, n_samples)
	xy = np.array([
		np.cos(t),
		np.sin(t)
	]).transpose()

	noise = create_noise(n_samples, 2, noise_factor)
	normalized_t = normalize_param_range(t, t_range)
	return (xy + noise, normalized_t)

def make_s_sheet(n_samples, noise_factor):
	# The s sheet is parameterized by 
	# x(s,t) = 0.5 + sin(t)cos(t)
	# y(s,t) = s
	# z(s,t) = 0.5 + 0.5cos(t)
	# for 0 <= s <= 1 and (3/4)pi <= t <= (9/4)pi

	s_range = (0, 1)
	t_range = (3.0 * np.pi / 4.0,
	           9.0 * np.pi / 4.0)
	s = np.random.uniform(*s_range, n_samples)
	t = np.random.uniform(*t_range, n_samples)
	xyz = np.array([
		0.5 + np.multiply(np.sin(t), np.cos(t)),
		s,
		0.5 + (0.5 * np.cos(t))
	]).transpose()

	noise = create_noise(n_samples, 3, noise_factor)
	normalized_s = normalize_param_range(s, s_range)
	normalized_t = normalize_param_range(t, t_range)
	params = np.array([normalized_s, normalized_t]).transpose()
	return (xyz + noise, params)

def make_sphere(n_samples, noise_factor, dimension=3):
	# By default, this will produce a 2-sphere in 3D, but can produce any
	# (n-1)-sphere in nD using the dimension argument. The uniform sample
	# is obtained by generating independent random normal variables, and
	# normalizing. See: http://mathworld.wolfram.com/SpherePointPicking.html

	xyz_raw = np.random.multivariate_normal(np.zeros(dimension), np.eye(dimension), n_samples)
	xyz = xyz_raw / np.linalg.norm(xyz_raw, axis=1).reshape(-1,1)

	# Inverse parameterization from https://mathworld.wolfram.com/Sphere.html
	phi = np.arccos(xyz[:,2])
	theta = np.arctan2(xyz[:,1], xyz[:,0])

	noise = create_noise(n_samples, dimension, noise_factor)
	normalized_phi = normalize_param_range(phi, (0, np.pi))
	normalized_theta = normalize_param_range(theta, (-np.pi, np.pi))
	params = np.array([normalized_theta, normalized_phi]).transpose()
	return (xyz + noise, params)

def make_swiss_roll(n_samples, noise_factor, b_val=0.05):
	# The swiss roll is parameterized by
	# x(s,t) = btcos(t)
	# y(s,t) = s
	# z(s,t) = btsin(t)
	# for 0 <= s <= 1.0 and 2pi <= t <= 6pi

	s_range = (0, 1)
	t_range = (2 * np.pi, 6 * np.pi)
	s = np.random.uniform(*s_range, n_samples)
	t = np.random.uniform(*t_range, n_samples)
	xyz = np.array([
		b_val * t * np.cos(t),
		s,
		b_val * t * np.sin(t)
	]).transpose()

	noise = create_noise(n_samples, 3, noise_factor)
	normalized_s = normalize_param_range(s, s_range)
	normalized_t = normalize_param_range(t, t_range)
	params = np.array([normalized_s, normalized_t]).transpose()
	return (xyz + noise, params)

def make_torus(n_samples, noise_factor, R=6, r=4):
	# The torus is parameterized by
	# x(theta, phi) = (R + rcos(theta)) * cos(phi)
	# y(theta, phi) = (R + rcos(theta)) * sin(phi)
	# z(theta, phi) = rsin(theta)
	# for 0 <= theta, phi <= 2pi

	theta_range = (0, 2*np.pi)
	phi_range = (0, 2*np.pi)
	theta = np.random.uniform(*theta_range, n_samples)
	phi = np.random.uniform(*phi_range, n_samples)
	xyz = np.array([
		(R + r * np.cos(theta)) * np.cos(phi),
		(R + r * np.cos(theta)) * np.sin(phi),
		r * np.sin(theta)
	]).transpose()

	noise = create_noise(n_samples, 3, noise_factor)
	normalized_theta = normalize_param_range(theta, theta_range)
	normalized_phi = normalize_param_range(phi, phi_range)
	params = np.array([normalized_theta, normalized_phi]).transpose()
	return (xyz + noise, params)

def make_cylinder(n_samples, noise_factor, r=1):
	# The cylinder is parameterized by
	# x(s, t) = rcos(t)
	# y(s, t) = rsin(t)
	# z(s, t) = s
	# for 0 <= s <= 1, 0 <= t <= 2pi

	s_range = (0, 1)
	t_range = (0, 2*np.pi)
	s = np.random.uniform(*s_range, n_samples)
	t = np.random.uniform(*t_range, n_samples)
	xyz = np.array([
		r * np.cos(t),
		r * np.sin(t),
		s
	]).transpose()

	noise = create_noise(n_samples, 3, noise_factor)
	normalized_s = normalize_param_range(s, s_range)
	normalized_t = normalize_param_range(t, t_range)
	params = np.array([normalized_s, normalized_t]).transpose()
	return (xyz + noise, params)

def make_mobius_strip(n_samples, noise_factor):
	# The Mobius strip is parameterized by
	# x(u, v) = (1 + vcos(u/2)/2)cos(u)
	# y(u, v) = (1 + vcos(u/2)/2)sin(u)
	# z(u, v) = vsin(u/2)/2
	# for 0 <= u <= 2pi, -1 <= v <= 1
	# See: https://en.wikipedia.org/wiki/M%C3%B6bius_strip

	u_range = (0, 2*np.pi)
	v_range = (-1, 1)
	u = np.random.uniform(*u_range, n_samples)
	v = np.random.uniform(*v_range, n_samples)
	xyz = np.array([
		(1 + v*np.cos(u/2)/2) * np.cos(u),
		(1 + v*np.cos(u/2)/2) * np.sin(u),
		v*np.sin(u/2)/2
	]).transpose()

	noise = create_noise(n_samples, 3, noise_factor)
	normalized_u = normalize_param_range(u, u_range)
	normalized_v = normalize_param_range(v, v_range)
	params = np.array([normalized_u, normalized_v]).transpose()
	return (xyz + noise, params)

def make_klein_bottle(n_samples, noise_factor, R=6, r=4):
	# The Klein botle is parameterized by
	# x(theta, phi) = (R + rcos(theta))cos(phi)
	# y(theta, phi) = (R + rcos(theta))sin(phi)
	# z(theta, phi) = rsin(theta)cos(phi/2)
	# w(theta, phi) = rsin(theta)sin(phi/2)
	# for 0 <= theta, phi <= 2pi
	# See: https://en.wikipedia.org/wiki/Klein_bottle#3D_pinched_torus_/_4D_M%C3%B6bius_tube

	theta_range = (0, 2*np.pi)
	phi_range = (0, 2*np.pi)
	theta = np.random.uniform(*theta_range, n_samples)
	phi = np.random.uniform(*phi_range, n_samples)
	xyzw = np.array([
		(R + r * np.cos(theta)) * np.cos(phi),
		(R + r * np.cos(theta)) * np.sin(phi),
		r * np.sin(theta) * np.cos(phi/2),
		r * np.sin(theta) * np.sin(phi/2)
	]).transpose()

	noise = create_noise(n_samples, 4, noise_factor)
	normalized_theta = normalize_param_range(theta, theta_range)
	normalized_phi = normalize_param_range(phi, phi_range)
	params = np.array([normalized_theta, normalized_phi]).transpose()
	return (xyzw + noise, params)

def make_SO3(n_samples, noise_factor):
	from scipy.stats import special_ortho_group
	from scipy.spatial.transform import Rotation

	mats = special_ortho_group.rvs(dim=3, size=n_samples)
	data = mats.reshape(-1,9)

	noise = create_noise(n_samples, 9, noise_factor)
	raw_params = Rotation.from_matrix(mats).as_euler(seq="xyz")

	x_range = (-np.pi, np.pi)
	y_range = (-np.pi/2, np.pi/2)
	z_range = (-np.pi, np.pi)

	x_params = normalize_param_range(raw_params[:,0], x_range)
	y_params = normalize_param_range(raw_params[:,1], y_range)
	z_params = normalize_param_range(raw_params[:,2], z_range)

	params = np.array([x_params, y_params, z_params]).T

	return (data + noise, params)

def make_RP2(n_samples, noise_factor):
	sphere_points, params = make_sphere(n_samples, noise_factor=0)

	flip_z_idx = sphere_points[:,2] > 0
	sphere_points[flip_z_idx,2] = sphere_points[flip_z_idx,2] * -1

	phi_range = (-np.pi, np.pi)
	raw_phi = (params[:,1] * (phi_range[1] - phi_range[0])) + phi_range[0]
	raw_phi[flip_z_idx] = -1 * raw_phi[flip_z_idx]
	new_phi_range = (0, np.pi)
	params[:,1] = normalize_param_range(raw_phi, new_phi_range)
	
	x, y, z = sphere_points[:,0], sphere_points[:,1], sphere_points[:,2]
	rp2_points = np.array([x*y, x*z, (y*y)-(z*z), 2*y*z]).T

	noise = create_noise(n_samples, 4, noise_factor)

	return (rp2_points + noise, params)