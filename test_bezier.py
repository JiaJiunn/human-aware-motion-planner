import pygame
import numpy as np


WINDOW_SIZE = 500
POINT_RADIUS = 3
BLACK = 0, 0, 0
WHITE = 255, 255, 255
RED = 255, 0, 0

POINT_LIST = [(50, 100), (100, 100), (100, 120), (110, 140), (170, 140), (200, 150), (200, 180), (200, 300), (250, 250), (300, 210), (370, 180), (400, 100), (450, 120)]
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


def main():

  # set up screen
  pygame.init()
  pygame.display.set_caption('test bezier curves')
  screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))

  init_curve = False
  while True:
    screen.fill(WHITE)
    pointies = []

    # draw points
    for idx in range(len(POINT_LIST)):
      p_x, p_y = POINT_LIST[idx]
      pygame.draw.circle(screen, BLACK, (p_x, p_y), POINT_RADIUS)

    if not init_curve:
      # generate bezier points
      for idx in range(len(POINT_LIST)-3):
        if idx % 3 == 0:
          add_curve(POINT_LIST[idx][0], POINT_LIST[idx][1], POINT_LIST[idx+1][0], POINT_LIST[idx+1][1], POINT_LIST[idx+2][0], POINT_LIST[idx+2][1], POINT_LIST[idx+3][0], POINT_LIST[idx+3][1], .001)
      init_curve = True

    # draw bezier points
    for j in range(len(BEZIER_POINTS)-1):
      pygame.draw.line(screen, RED, (BEZIER_POINTS[j][0], BEZIER_POINTS[j][1]), (BEZIER_POINTS[j+1][0], BEZIER_POINTS[j+1][1]))
    pygame.display.update()


if __name__ == "__main__":
    main()