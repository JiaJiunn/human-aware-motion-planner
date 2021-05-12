from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt
import pygame
import numpy as np
import scipy.stats
from skimage.draw import line
import yaml
from a_star import astar
from collections import deque

PARENT_DIR = Path(__file__).parent.parent
HUMAN_RADIUS = 3
HUMAN_DIR_LEN = 10
HUMAN_DIR_ANGLE_LEN = 15
HUMAN_DIR_ANGLE_LEN_BLOCKED = 30
STANDING_SAFETY = 13 # NOTE MUST BE ODD "less threatened when standing"
STANDING_KERNEL_SIZE = 3
SITTING_SAFETY = 21 # NOTE MUST BE ODD
SITTING_KERNEL_SIZE = 3
VISIBILITY_RATIO = -1
AWARENESS_KERNEL_SIZE = 1
INTERRUPT_COST_RATIO = 0.7
BLACK = 0, 0, 0
WHITE = 255, 255, 255
RED = 255, 0, 0
BLUE = 0, 255, 0
BEZIER_POINTS = []


def generate_curve_coefs(p1, p2, p3, p4):
	m2 = np.array([p1, p2, p3, p4])
	m1 = np.array([[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
	return (m1 @ m2)


def add_curve(x0, y0, x1, y1, x2, y2, x3, y3, step):
	t = 0
	while (t < 1):
		x_matrix = generate_curve_coefs(x0, x1, x2, x3)
		y_matrix = generate_curve_coefs(y0, y1, y2, y3)
		x = x_matrix[0] * (t ** 3) + x_matrix[1] * (t ** 2) + x_matrix[2] * t + x_matrix[3]
		y = y_matrix[0] * (t ** 3) + y_matrix[1] * (t ** 2) + y_matrix[2] * t + y_matrix[3]
		BEZIER_POINTS.append([x, y])
		t += step


def euc_dist(point1, point2):
	point1_np = np.array(point1)
	point2_np = np.array(point2)
	return np.linalg.norm(point2_np - point1_np)


def weighted_euc_dist(point1, point2, point3):
	point1_np = np.array(point1)
	point2_np = np.array(point2)
	point3_np = np.array(point3)
	# import pdb; pdb.set_trace()
	if not np.any(point2_np - point1_np):
		return np.linalg.norm(point3_np - point2_np)
	else: 
		prev_vec = (point2_np - point1_np)/np.linalg.norm(point2_np - point1_np)
		curr_vec = (point3_np - point2_np)/np.linalg.norm(point3_np - point2_np)
		return np.dot(curr_vec, prev_vec) * np.linalg.norm(point3_np - point2_np)


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
			map_mat_copy[rr, cc] = get_normal_red(len(map_mat[rr, cc]), 1) * VISIBILITY_RATIO

		map_mat += map_mat_copy # important for overlaps
		
	return map_mat


def get_hidden_zone_cost(map_dict):
	map_size = map_dict['map_size']
	map_mat = get_obs_cost(map_dict)
	map_mat_empty = np.zeros((map_size, map_size))

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
			x_diff, y_diff = round(x_diff * HUMAN_DIR_ANGLE_LEN_BLOCKED, 2), round(y_diff * HUMAN_DIR_ANGLE_LEN_BLOCKED, 2)
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
			vis_arr = np.zeros(len(rr))
			vis_blocked = False
			normal_red = get_normal_red(len(rr), AWARENESS_KERNEL_SIZE) # TODO
			for idx in range(len(rr)):
				if map_mat[rr[idx], cc[idx]] != 0:
					vis_blocked = True
				if vis_blocked:
					vis_arr[idx] = normal_red[idx]
			map_mat_copy[rr, cc] = vis_arr

		map_mat_empty += map_mat_copy # important for overlaps

	return map_mat_empty


# def euc_dist(point1, point2):
#     return np.linalg.norm(point1 - point2)


def get_interrupt_cost(map_dict):
    
    map_mat = np.zeros((map_dict['map_size'], map_dict['map_size']))
#     map_mat = get_obs_cost(map_dict)
    
    for interaction in map_dict['interactions']:
        obs_idx, hum_idx = interaction['indices']

        obs = map_dict['obstacles'][obs_idx]
        hum = map_dict['humans'][hum_idx]

        hum_x, hum_y = hum['coordinates']
        tl_x, tl_y, br_x, br_y = obs['coordinates']

        # get two closest edges of object
        obs_edges = np.array([[tl_x, tl_y], [br_x, tl_y], [tl_x, br_y], [br_x, br_y]])
        euc_dist_lst = []
        for edge_coord in obs_edges:
            euc_dist_lst.append(euc_dist((hum_x, hum_y), edge_coord))
        first_edge_idx, second_edge_idx = np.argpartition(euc_dist_lst, 2)[:2]

        dir_vec_1 = np.array([obs_edges[first_edge_idx][1] - hum_y, obs_edges[first_edge_idx][0] - hum_x])
        dir_vec_1_normalized = dir_vec_1/np.linalg.norm(dir_vec_1)
        deg_1 = np.degrees(np.arctan2(dir_vec_1_normalized[0], dir_vec_1_normalized[1])) + 1
        if deg_1 < 0:
            deg_1 += 360

        dir_vec_2 = np.array([obs_edges[second_edge_idx][1] - hum_y, obs_edges[second_edge_idx][0] - hum_x])
        dir_vec_2_normalized = dir_vec_2/np.linalg.norm(dir_vec_2)
        deg_2 = np.degrees(np.arctan2(dir_vec_2_normalized[0], dir_vec_2_normalized[1])) + 1
        if deg_2 < 0:
            deg_2 += 360

        # TODO degree edge case
        if deg_2 < deg_1:
            deg_1, deg_2 = deg_2, deg_1
            
        interrupt_len = max(np.linalg.norm(dir_vec_1), np.linalg.norm(dir_vec_2))
        for angle in range(int(deg_1), int(deg_2)):
            x_diff, y_diff = angle_to_vec(angle)
            x_diff, y_diff = round(x_diff * interrupt_len, 2), round(y_diff * interrupt_len, 2)
            rr, cc = line(hum_y, hum_x, int(hum_y+y_diff), int(hum_x+x_diff))

            # checks if cells are all within grid
            in_bounds = []
            for idx in range(len(rr)):
                if rr[idx] >= 0 and rr[idx] < map_dict['map_size'] and cc[idx] >= 0 and cc[idx] < map_dict['map_size']:
                    in_bounds.append(True)
                else:
                    in_bounds.append(False)
            rr = rr[in_bounds]
            cc = cc[in_bounds]

            map_mat[rr, cc] = np.ones(len(map_mat[rr, cc])) * INTERRUPT_COST_RATIO
    
    return map_mat


# for visualizations
def overlay_obs_hum(map_dict, cost_map):

	for obs in map_dict['obstacles']:
		tl_x, tl_y, br_x, br_y = obs['coordinates']
		obs_shape = (br_y-tl_y+1, br_x-tl_x+1)
		cost_map[tl_y:(br_y+1), tl_x:(br_x+1)] = np.ones(obs_shape) * 100

	for hum in map_dict['humans']:
		x, y = hum['coordinates']
		cost_map[y, x] = 100

	return cost_map


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

	# set up cost map
	test_safety_cost = get_safety_cost(map_dict)
	test_visibility_cost = get_visibility_cost(map_dict)
	test_obs_cost = get_obs_cost(map_dict) * 500
	test_hidden_zone_cost = get_hidden_zone_cost(map_dict) * 20
	test_interrupt_cost = get_interrupt_cost(map_dict) * 10
	cost_map = test_safety_cost + test_visibility_cost + test_obs_cost + test_hidden_zone_cost + test_interrupt_cost
	cost_map *= 15
	cost_map += 15

	# hard coded path TODO
	path = astar(cost_map, (90, 50), (10, 90))
	# path = astar(cost_map, (10, 50), (10, 90))

	# display
	display_map = overlay_obs_hum(map_dict, cost_map)
	for coor_x, coor_y in path:
		display_map[coor_x, coor_y] = 100
	plt.imshow(display_map, cmap='gray')
	plt.show()
	import pdb; pdb.set_trace()

	for i in range(len(path)):
		path[i] = (path[i][1] * screen_resolution, path[i][0] * screen_resolution)
	
	init_curve = False
	BOT_IDX = 0
	PREV_IDX = deque()
	PREV_IDX.append(0)
	while True:
		screen.fill(WHITE)

		if not init_curve:
			# generate bezier points
			for idx in range(len(path)-12):
				if idx % 12 == 0:
					add_curve(path[idx][0], path[idx][1], path[idx+4][0], path[idx+4][1], path[idx+8][0], path[idx+8][1], path[idx+12][0], path[idx+12][1], .0001)
			init_curve = True
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
		for j in range(len(path)-1):
			pygame.draw.line(screen, RED, (path[j][0], path[j][1]), (path[j+1][0], path[j+1][1]))
		
		# draw bot (constant speed)
		pygame.draw.circle(screen, BLUE, BEZIER_POINTS[BOT_IDX], 3)
		next_bot_idx = BOT_IDX + 1
		while euc_dist(BEZIER_POINTS[BOT_IDX], BEZIER_POINTS[next_bot_idx]) < 0.1:
			next_bot_idx += 10
			if next_bot_idx >= len(BEZIER_POINTS):
				next_bot_idx = 0
				break
		BOT_IDX = next_bot_idx
		import pdb; pdb.set_trace()

		# draw bot (slow when turn)
		# pygame.draw.circle(screen, BLUE, BEZIER_POINTS[BOT_IDX], 3)
		# next_bot_idx = BOT_IDX + 1
		# while weighted_euc_dist(BEZIER_POINTS[PREV_IDX[-1]], BEZIER_POINTS[BOT_IDX], BEZIER_POINTS[next_bot_idx]) < 1:
		# 	next_bot_idx += 10
		# 	if next_bot_idx >= len(BEZIER_POINTS):
		# 		next_bot_idx = 0
		# 		PREV_IDX.clear()
		# 		PREV_IDX.append(0)
		# 		break
		# PREV_IDX.append(BOT_IDX)
		# if len(PREV_IDX) == 5:
		# 	PREV_IDX.pop()
		# BOT_IDX = next_bot_idx

		pygame.display.update()


if __name__ == "__main__":
    main()
