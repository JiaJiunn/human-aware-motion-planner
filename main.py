import pygame
import yaml


FILE_PATH = './test_map.yaml'
BLACK = 0, 0, 0
WHITE = 255, 255, 255

def main():
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    tl_x, tl_y, br_x, br_y = [20, 40, 30, 20]
    width_height = (br_x - tl_x, tl_y - br_y)

    screen.fill(WHITE)
    while True:
        rect = pygame.Rect((tl_x, tl_y), width_height)
        pygame.draw.rect(screen, BLACK, rect)
        pygame.display.update()

if __name__ == "__main__":
  main()