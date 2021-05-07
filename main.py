from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt
import pygame
import numpy as np
import scipy.stats
from skimage.draw import line
import yaml


PARENT_DIR = Path(__file__).parent.parent
HUMAN_RADIUS = 3
HUMAN_DIR_LEN = 10
HUMAN_DIR_ANGLE_LEN = 15
STANDING_SAFETY = 13 # NOTE MUST BE ODD "less threatened when standing"
STANDING_KERNEL_SIZE = 3
SITTING_SAFETY = 21 # NOTE MUST BE ODD
SITTING_KERNEL_SIZE = 3
BLACK = 0, 0, 0
WHITE = 255, 255, 255
RED = 255, 0, 0

def arg_parser():
    """Returns parser for cli arguments."""

    parser = argparse.ArgumentParser(description="motion planner arguments")
    parser.add_argument("--map", type=str, required=True)

    return parser


def parse_yaml(yaml_path):
    """Parses yaml into dictionaries."""

    if not yaml_path.exists():
        raise NotImplementedError("{} not found".format(yaml_path))
    with open(yaml_path) as file:
        yaml_dict = yaml.safe_load(file)

    return yaml_dict


def angle_to_vec(angle_degrees):
	angle_radians = math.radians(angle_degrees)
	x_diff = math.cos(angle_radians)
	y_diff = math.sin(angle_radians)
	return x_diff, y_diff


def gaussian_kernel(kernlen, nsig):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def gaussian_kernel_normalized(kernlen, nsig):
    """Returns a normalized 2D Gaussian kernel."""

    kernel = gaussian_kernel(kernlen, nsig)
    return (kernel - np.min(kernel))/(np.max(kernel) - np.min(kernel))


def get_normal_red(len_dist, n_sig):
    kernlen = (len_dist * 2)
    x = np.linspace(-n_sig, n_sig, kernlen+1)
    kern_1d = np.diff(scipy.stats.norm.cdf(x))
    norm_red = kern_1d[len_dist:]
    normalized_norm_red = (norm_red - np.min(norm_red))/(np.max(norm_red) - np.min(norm_red))
    return normalized_norm_red


def hum_safety(map_dict, normal_mat, coor):
	"""Returns the human safety matrix."""
	map_mat_copy = np.zeros((map_dict['map_size'], map_dict['map_size']))
	normal_mat_radius = normal_mat.shape[0] // 2
	x_coor, y_coor = coor

	# slice the start
	x_exc = x_coor - normal_mat_radius
	y_exc = y_coor - normal_mat_radius
	x_trunc = 0
	y_trunc = 0
	if x_exc < 0:
		x_trunc = abs(x_exc)
	if y_exc < 0:
		y_trunc = abs(y_exc)

	# slice the end
	x_exc2 = x_coor + 1 + normal_mat_radius - map_mat_copy.shape[0]
	y_exc2 = y_coor + 1 + normal_mat_radius - map_mat_copy.shape[1]
	x_trunc2 = normal_mat.shape[0]
	y_trunc2 = normal_mat.shape[1]
	if x_exc2 > 0:
		x_trunc2 -= x_exc2
	if y_exc2 > 0:
		y_trunc2 -= y_exc2

	# get truncated normal mat
	trunc_normal_mat = normal_mat[x_trunc:x_trunc2, y_trunc:y_trunc2]

	xl_idx = max(x_coor - normal_mat_radius, 0)
	xr_idx = min(x_coor + normal_mat_radius, map_mat_copy.shape[0])
	yb_idx = max(y_coor - normal_mat_radius, 0)
	yt_idx = min(y_coor + normal_mat_radius, map_mat_copy.shape[1])

	map_mat_copy[xl_idx:(xr_idx+1), yb_idx:(yt_idx+1)] = trunc_normal_mat

	return map_mat_copy


def get_obs_cost(map_dict):
	map_mat = np.zeros((map_dict['map_size'], map_dict['map_size']))

	for obs in map_dict['obstacles']:
		tl_x, tl_y, br_x, br_y = obs['coordinates']
		obs_shape = (br_y-tl_y+1, br_x-tl_x+1)
		map_mat[tl_y:(br_y+1), tl_x:(br_x+1)] = np.ones(obs_shape)

	return map_mat


def get_safety_cost(map_dict):
	map_mat = np.zeros((map_dict['map_size'], map_dict['map_size']))

	for hum in map_dict['humans']:

		x, y = hum['coordinates']

		# human safety level dependant on human state
		if hum['state'] == "SITTING":
			normal_mat = gaussian_kernel_normalized(SITTING_SAFETY, SITTING_KERNEL_SIZE)
		elif hum['state'] == "STANDING":
			normal_mat = gaussian_kernel_normalized(STANDING_SAFETY, STANDING_KERNEL_SIZE)
		else:
			raise Exception('Human state not defined.')

		ret_safety_mat = hum_safety(map_dict, normal_mat, (y, x)) # matrix x y flipped
		map_mat = np.add(map_mat, ret_safety_mat)

	return map_mat


def get_visibility_cost(map_dict): # TODO invert costs (is currently reward, *-1?)
	map_size = map_dict['map_size']
	map_mat = np.zeros((map_size, map_size))

	for hum in map_dict['humans']:
		x, y = hum['coordinates']
		
		dir_angle = hum['direction']
		map_mat_copy = np.zeros((map_size, map_size))
		for angle in range(dir_angle-45, dir_angle+45):

			# gets angle
			if angle >= 360:
				angle -= 360
			if angle < 0:
				angle += 360
					
			# gets line from angle
			x_diff, y_diff = angle_to_vec(angle)
			x_diff, y_diff = round(x_diff * HUMAN_DIR_ANGLE_LEN, 2), round(y_diff * HUMAN_DIR_ANGLE_LEN, 2)
			rr, cc = line(y, x, int(y+y_diff), int(x+x_diff))
			
			# checks if cells are all within grid
			in_bounds = []
			for idx in range(len(rr)):
				if rr[idx] >= 0 and rr[idx] < map_size and cc[idx] >= 0 and cc[idx] < map_size:
					in_bounds.append(True)
				else:
					in_bounds.append(False)
			rr = rr[in_bounds]
			cc = cc[in_bounds]
			
			# TODO check if vector has been plotted before
			map_mat_copy[rr, cc] = get_normal_red(len(map_mat[rr, cc]), 1)

		map_mat += map_mat_copy # important for overlaps
		
	return map_mat


def main():

	# parse args
	parser = arg_parser()
	args = parser.parse_args()

	# get map configs
	map_path = PARENT_DIR / args.map
	map_dict = parse_yaml(map_path)

	# set up screen
	pygame.init()
	pygame.display.set_caption(map_dict['title'])
	screen_resolution = map_dict['resolution']
	window_size = map_dict['map_size'] * screen_resolution
	screen = pygame.display.set_mode((window_size, window_size))

	# test_safety_cost = get_safety_cost(map_dict)
	# test_visibility_cost = get_visibility_cost(map_dict)
	# test_obs_cost = get_obs_cost(map_dict)
	# plt.imshow(test_obs_cost, cmap='gray')
	# plt.show()

	while True:
		screen.fill(WHITE)

		# set up obstacles
		for obs in map_dict['obstacles']:

			tl_x, tl_y, br_x, br_y = obs['coordinates']
			# update according to resolution
			obs_shape = ((br_x - tl_x) * screen_resolution, (br_y - tl_y) * screen_resolution)
			rect = pygame.Rect((tl_x * screen_resolution, tl_y * screen_resolution), obs_shape)
			pygame.draw.rect(screen, BLACK, rect)
		
		# set up humans
		for hum in map_dict['humans']:

			# location
			x, y = hum['coordinates']
			x *= screen_resolution
			y *= screen_resolution
			pygame.draw.circle(screen, RED, (x, y), HUMAN_RADIUS)

			# facing direction
			x_diff, y_diff = angle_to_vec(hum['direction'])
			x_diff *= HUMAN_DIR_LEN
			y_diff *= HUMAN_DIR_LEN
			pygame.draw.line(screen, RED, (x, y) , (x + x_diff, y + y_diff), width=HUMAN_RADIUS)

		pygame.display.update()


if __name__ == "__main__":
    main()
