from pathlib import Path
import argparse
import math
import pygame
import numpy as np
import scipy.stats
import yaml


PARENT_DIR = Path(__file__).parent.parent
HUMAN_RADIUS = 3
HUMAN_DIR_LEN = 10
STANDING_COMF = 1
SITTING_COMF = 3
FIXED_KERNEL_SIZE = 10
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


def gkern(kernlen, nsig):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(scipy.stats.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def get_safety_cost(map_dict):
	map_mat = np.zeros((map_dict['map_size'], map_dict['map_size']))
	# for obs in map_dict['obstacles']:
	# 	tl_x, tl_y, br_x, br_y = obs['coordinates']
	# 	obs_shape = (br_y-tl_y+1, br_x-tl_x+1)
	# 	map_mat[tl_y:(br_y+1), tl_x:(br_x+1)] = np.ones(obs_shape)

	# TODO add costs
	# if sitting, use gkern(FIXED_KERNEL_SIZE, SITTING_COMF) # slice then add
	# if standing, use gkern(FIXED_KERNEL_SIZE, STANDING_COMF)

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
