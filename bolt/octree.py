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
        self.num_points = 0
        self.parent = None

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, OctreeNode):
            return False

        if self.point is None and __value.point is not None:
            return False

        if self.point is not None and __value.point is None:
            return False

        return np.array_equal(self.point, __value.point) and\
            np.array_equal(self.center, __value.center)


class Octree():
    def __init__(self):
        self.tree = None
        self.max_depth = 64

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
            node.children[direction].parent = node
        node.is_leaf = False

    def exists(self, node, point):
        if np.array_equal(node.point, point) or np.linalg.norm(node.center - point) < 1e-6:
            return True
        if node.is_leaf:
            return False
        direction = self.get_direction(point, node.center)
        return self.exists(node.children[direction], point)

    def insert(self, node, point):
        if self.exists(node, point):
            return

        node.num_points += 1

        if node.is_leaf:
            if node.point is None:
                node.point = point
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
        points = np.array(points)
        return points

    def _child_has_num_points(self, node, k):
        return any([child.num_points >= k for child in node.children.values()])

    def _child_with_num_points(self, node, k):
        for child in node.children.values():
            if child.num_points >= k:
                return child
        return None
    
    def knn_points(self, point, points, k):
        indices = np.argsort(np.linalg.norm(point - points, axis=1))[:k]
        return points[indices]

    def get_k_nearest_neighbours(self, node, point, k):
        direction = self.get_direction(point, node.center)
        if node.num_points == k or\
                direction not in node.children or\
                (node.num_points > k and node.children[direction].num_points < k):
             points = self.get_points(node)
             return self.knn_points(point, np.array(points), k)
        else:
            return self.get_k_nearest_neighbours(node.children[direction], point, k)

    def get_height(self, node):
        if node.is_leaf:
            return 0
        return 1 + max([self.get_height(child) for child in node.children.values()])
