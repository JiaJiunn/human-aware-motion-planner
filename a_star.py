import numpy as np
import heapq

class Node():
    def __init__(self, backpointer, position):
        self.backpointer = backpointer
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __gt__(self, other):
        return self.f > other.f

    def __lt__(self, other):
        return self.f < other.f

    def __str__(self):
        return str(self.position)

    def __hash__(self):
        return hash(str(self))


# Returns list of pairs of coordinates on path
# start and end are integer pairs
# maze can be either list of lists or np.array
def astar(maze, start, end):
    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)


    # Initialize both open and closed list
    open_list = []
    closed_list = set()

    # Add the start node
    heapq.heappush(open_list, start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # for node in open_list:
        #    print(str(node) + " ")
        #print('\n')
        # Get the current node
        current_node = heapq.heappop(open_list)

        # print("current node: ", current_node)

        # Pop current off open list, add to closed list
        closed_list.add(current_node)
        
        # Found the goal
        if current_node == end_node:
            # print('FOUND GOAL')
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.backpointer
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            if child in closed_list:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + maze[child.position[0]][child.position[1]] + 1

            # This estimated distance from goal is garbage cuz it's hard to estimate. This def is slowing down astar 
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            # This is slowing down alg too, should switch this to heep
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            heapq.heappush(open_list, child)


def main():

    maze1 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [10000, 10000, 100, 100, 100, 100, 100, 100, 100, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            [1, 100, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 100, 1, 100, 100, 100, 100, 100, 100, 1],
            [1, 100, 1, 100, 100, 100, 100, 100, 100, 1],
            [1, 100, 1, 1, 1, 1, 1, 1, 100, 1],
            [1, 100, 100, 100, 100, 100, 100, 1, 100, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 100, 1]
    ])





    start = (0, 0)
    end = (9,9)

    path = astar(maze1, start, end)
    print(path)

# main()