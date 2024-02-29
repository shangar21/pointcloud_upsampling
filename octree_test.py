import octree
import numpy as np

def octreenode_create_test():
    # Create an octree node
    point = np.array([0, 0, 0])
    center = np.array([0, 0, 0])
    node = octree.OctreeNode(point, center)
    assert node is not None
    assert np.array_equal(node.point, point)    
    assert np.array_equal(node.center, center)

def octree_create_test():
    # Create an octree
    oct = octree.Octree()
    assert oct is not None
    assert oct.tree is None

def create_root_test():
    # Create a root node
    oct = octree.Octree()
    points = np.array([[0, 0, 0], [1, 1, 1]])
    oct._create_root(points)
    assert np.array_equal(oct.tree.center, [0.5, 0.5, 0.5])
    assert oct.tree.width == 0.5
    assert oct.tree.height == 0.5
    assert oct.tree.depth == 0.5
    assert oct.tree.is_leaf == True

def get_direction_test():
    # Get the direction of a point
    oct = octree.Octree()
    point = np.array([-1, -1, -1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 0 
    point = np.array([-1, -1, 1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 1
    point = np.array([-1, 1, -1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 2
    point = np.array([-1, 1, 1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 3
    point = np.array([1, -1, -1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 4
    point = np.array([1, -1, 1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 5
    point = np.array([1, 1, -1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 6
    point = np.array([1, 1, 1])
    center = np.array([0, 0, 0])
    direction = oct.get_direction(point, center)
    assert direction == 7

def subdivide_test():
    # Subdivide a node
    oct = octree.Octree()
    points = np.array([[0, 0, 0], [1, 1, 1]])
    oct._create_root(points)
    # subdivide the root node
    oct.subdivide(oct.tree)
    for i in oct.tree.children.values():
        assert i.width == 0.25
        assert i.height == 0.25
        assert i.depth == 0.25
        assert (np.abs(i.center - oct.tree.center) - np.array([0.25, 0.25, 0.25])).sum() == 0
    assert oct.tree.is_leaf == False

    # subdivide a child node
    oct.subdivide(oct.tree.children[1])
    for i in oct.tree.children[1].children.values():
        assert i.width == 0.125
        assert i.height == 0.125
        assert i.depth == 0.125
        assert (np.abs(i.center - oct.tree.children[1].center) - np.array([0.125, 0.125, 0.125])).sum() == 0
    assert oct.tree.children[1].is_leaf == False

def insert_test():
    # Insert a point into the octree
    oct = octree.Octree()
    points = np.array([[0, 0, 0], [1, 1, 1]])
    oct._create_root(points)
    oct.insert(oct.tree, points[0])
    oct.insert(oct.tree, points[1])
    assert oct.tree.is_leaf == False
    assert oct.tree.children[0].is_leaf == True
    assert oct.tree.children[0].point is not None
    assert np.array_equal(oct.tree.children[0].point, points[0])
    assert oct.tree.children[7].is_leaf == True
    assert oct.tree.children[7].point is not None
    assert np.array_equal(oct.tree.children[7].point, points[1])

    assert oct.tree.children[1].point is None
    assert oct.tree.children[2].point is None
    assert oct.tree.children[3].point is None
    assert oct.tree.children[4].point is None
    assert oct.tree.children[5].point is None
    assert oct.tree.children[6].point is None

def insert_duplicate_test():
    oct = octree.Octree()
    points = np.array([[0, 0, 0], [1, 1, 1]])
    oct._create_root(points)
    oct.insert(oct.tree, points[0])
    oct.insert(oct.tree, points[0])
    assert not oct.tree.children
    assert np.array_equal(oct.tree.point, points[0])

    oct.insert(oct.tree, points[1])
    oct.insert(oct.tree, points[1])

    assert oct.tree.is_leaf == False
    assert oct.tree.children[0].is_leaf == True
    assert oct.tree.children[0].point is not None
    assert np.array_equal(oct.tree.children[0].point, points[0])
    assert oct.tree.children[7].is_leaf == True
    assert oct.tree.children[7].point is not None
    assert np.array_equal(oct.tree.children[7].point, points[1])


def create_test():
    oct = octree.Octree()
    points = np.array([[0, 0, 0], [1, 1, 1]])
    oct.create(points)
    assert oct.tree.is_leaf == False
    assert oct.tree.children[0].is_leaf == True
    assert oct.tree.children[0].point is not None
    assert np.array_equal(oct.tree.children[0].point, points[0])
    assert oct.tree.children[7].is_leaf == True
    assert oct.tree.children[7].point is not None
    assert np.array_equal(oct.tree.children[7].point, points[1])

    assert oct.tree.children[1].point is None
    assert oct.tree.children[2].point is None
    assert oct.tree.children[3].point is None
    assert oct.tree.children[4].point is None
    assert oct.tree.children[5].point is None
    assert oct.tree.children[6].point is None

    

if __name__ == "__main__":
    octreenode_create_test()
    octree_create_test()
    create_root_test()
    get_direction_test()
    subdivide_test()
    insert_test()
    insert_duplicate_test()
    create_test()
    print("All tests pass")
