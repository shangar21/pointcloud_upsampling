from collections import deque
import numpy as np

DIRECTION_TO_COORDINATES = {
    0: [-1, -1, -1],
    1: [-1, -1, 1],
    2: [-1, 1, -1],
    3: [-1, 1, 1],
    4: [1, -1, -1],
    5: [1, -1, 1],
    6: [1, 1, -1],
    7: [1, 1, 1]
}

class OctreeNode():
    def __init__(self, point, center):
        self.point = point
        self.children = {}
        self.width = 0
        self.height = 0
        self.depth = 0
        self.center = center 
        self.is_leaf = True

class Octree():
    def __init__(self):
        self.tree = None

    def get_direction(self, point, center):
        direction = 0
        if point[0] > center[0]:
            direction |= 4
        if point[1] > center[1]:
            direction |= 2
        if point[2] > center[2]:
            direction |= 1
        return direction

    def subdivide(self, node):
        for direction, coordinates in DIRECTION_TO_COORDINATES.items():
            mags = np.array([node.width, node.height, node.depth])
            new_center = node.center + np.multiply(mags / 2, coordinates)
            node.children[direction] = OctreeNode(None, new_center)
            node.children[direction].width = node.width / 2
            node.children[direction].height = node.height / 2
            node.children[direction].depth = node.depth / 2
        node.is_leaf = False

    def insert(self, node, point):
        if node.is_leaf:
            if node.point is None:
                node.point = point
                return
            elif np.array_equal(node.point, point):
                return
            else:
                self.subdivide(node)
                direction = self.get_direction(node.point, node.center)
                self.insert(node.children[direction], node.point)
                direction = self.get_direction(point, node.center)
                self.insert(node.children[direction], point)
            node.point = None
        else:
            direction = self.get_direction(point, node.center)
            self.insert(node.children[direction], point)
        
    def _create_root(self, points):
        self.tree = [-1] * (len(points) + 1)
        center = np.mean(points, axis=0)
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        depth = np.max(points[:, 2]) - np.min(points[:, 2])
        self.tree = OctreeNode(None, center)
        self.tree.width = width / 2
        self.tree.height = height / 2
        self.tree.depth = depth / 2
        
    def create(self, points):
        self._create_root(points)
        for point in points:
            self.insert(self.tree, point)

    def deepest_parent(self, point):
        has_child = False
        node = self.tree
        while not has_child:
            direction = self.get_direction(point, node.center)
            if node.children and node.children[direction].is_leaf and\
                    np.array_equal(node.children[direction].point, point):
                has_child = True
                return node
            elif not node.children:
                has_child = False
            else:
                node = node.children[direction]

    def get_points(self, node=None):
        if node is None:
            node = self.tree
        points = []
        if node.is_leaf:
            if node.point is not None:
                points.append(node.point)
        else:
            for child in node.children.values():
                points.extend(self.get_points(child))
        return points
