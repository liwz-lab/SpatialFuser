import numpy as np
import torch
import sklearn.neighbors
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from functools import partial
from typing import List, Mapping, Optional, Union
import cv2
import time
import sys
import sklearn.neighbors
import scipy.optimize
from scipy.spatial import Delaunay


def apply_transformation(points, transform):
    R = np.array([[np.cos(transform[2]), -np.sin(transform[2])],
                  [np.sin(transform[2]), np.cos(transform[2])]])
    return points @ R.T + transform[:2]


def plot_results(src_points, dst_points, src_points_transformed, method:str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Source Points
    axes[0].scatter(src_points[:, 0], src_points[:, 1], c='r')
    axes[0].set_title('Source Points')
    axes[0].axis('equal')

    # Target Points
    axes[1].scatter(dst_points[:, 0], dst_points[:, 1], c='b')
    axes[1].set_title('Target Points')
    axes[1].axis('equal')

    # Transformed Source Points
    axes[2].scatter(src_points_transformed[:, 0], src_points_transformed[:, 1], c='g')
    axes[2].set_title('Transformed Source Points')
    axes[2].axis('equal')

    plt.suptitle('2D '+method+' Scan Matching')
    plt.tight_layout()
    plt.show()


class Grid:
    def __init__(self):
        self.points = []
        self.mean = 0
        self.cov_inv = np.zeros((2, 2), dtype=np.float32)

    def add_point(self, point):
        self.points.append(point)

    def compute_mean_cov(self):
        self.mean = np.mean(self.points, axis=0)
        cov = np.cov(np.transpose(self.points))
        epsilon = 1e-5
        cov += epsilon * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.inv(cov)

    def compute_score(self, x, y):
        score = np.exp(-0.1 * np.dot(np.dot(np.array([x, y]) - self.mean, self.cov_inv),
                                     np.transpose(np.array([x, y]) - self.mean)))
        return score


class Grids:
    def __init__(self, dst_points, grid_resolution):
        '''
        :param dst_points: coordinate data to grid
        :param grid_resolution: resolution controls the size of grids
        '''
        self.dst_points = dst_points
        self.grid_resolution = grid_resolution
        self.grids = self.create_grids()

    def create_grids(self):
        self.grid_min_x = np.min(self.dst_points[:, 0])
        self.grid_max_x = np.max(self.dst_points[:, 0])
        self.grid_min_y = np.min(self.dst_points[:, 1])
        self.grid_max_y = np.max(self.dst_points[:, 1])

        self.grid_size_x = int(np.ceil((self.grid_max_x - self.grid_min_x) / self.grid_resolution))
        self.grid_size_y = int(np.ceil((self.grid_max_y - self.grid_min_y) / self.grid_resolution))

        grids = np.empty((self.grid_size_x, self.grid_size_y), dtype=object)

        for i in range(len(self.dst_points)):
            x = min(max(int((self.dst_points[i, 0] - self.grid_min_x) / self.grid_resolution), 0), self.grid_size_x - 1)
            y = min(max(int((self.dst_points[i, 1] - self.grid_min_y) / self.grid_resolution), 0), self.grid_size_y - 1)
            if grids[x, y] is None:
                grids[x, y] = Grid()
            grids[x, y].add_point(self.dst_points[i])

        for i in range(self.grid_size_x):
            for j in range(self.grid_size_y):
                if grids[i, j] is not None and len(grids[i, j].points) > 2:
                    grids[i, j].compute_mean_cov()
        return grids

    def fit_transformed_data(self, transformed_data):
        src_grid_index = np.empty(transformed_data.shape)
        for i in range(src_grid_index.shape[0]):
            src_grid_index[i, 0] = min(max(int((transformed_data[i, 0] - self.grid_min_x) / self.grid_resolution), 0),
                                       self.grid_size_x - 1)
            src_grid_index[i, 1] = min(max(int((transformed_data[i, 1] - self.grid_min_y) / self.grid_resolution), 0),
                                       self.grid_size_y - 1)
        return src_grid_index.astype(int)

    def show_score_distribution(self):
        grid_size_x = self.grids.shape[0]
        grid_size_y = self.grids.shape[1]
        grid_scores = np.zeros((grid_size_x, grid_size_y), dtype=np.float32)
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                if self.grids[i, j] is not None and len(self.grids[i, j].points) > 2:
                    for point in self.grids[i, j].points:
                        grid_scores[i, j] += self.grids[i, j].compute_score(point[0], point[1])
                else:
                    grid_scores[i, j] = 0

        plt.figure(figsize=(5, 5))
        plt.imshow(grid_scores, origin='lower')
        plt.show()


def compute_grids_score(params, src_points, dst_grids):
    transformed_points = apply_transformation(src_points, params)
    transformed_index = dst_grids.fit_transformed_data(transformed_points)
    score = 0
    dst_grids_shape = dst_grids.grids.shape
    for i in range(src_points.shape[0]):
        point = transformed_points[i]
        index = transformed_index[i]
        if index[0] >= 0 and index[0] < dst_grids_shape[0] and index[1] >= 0 and index[1] < dst_grids_shape[1]:
            if dst_grids.grids[index[0], index[1]] != None:
                score += dst_grids.grids[index[0], index[1]].compute_score(point[0], point[1])

    return -score


def icp(a, b,
        max_time: Optional[int] = 4
        ):
    """
    Iterative Closest Point (ICP) algorithm

    Parameters
    ----------
    a
        dataset a, whose shape is [n,2]
    b
        dataset a, whose shape is [m,2]
    max_time
        max iter times

    Return
    ----------
    T_opt
        optimal transform matrix, please see https://docs.opencv.org/3.4/da/d6e/tutorial_py_geometric_transformations.html
    error_max
        error

    Refer
    ----------
    https://stackoverflow.com/questions/20120384/iterative-closest-point-icp-implementation-on-python
    """

    def res(p, src, dst):
        T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                          [np.sin(p[2]), np.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = np.size(src, 0)
        xt = np.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = np.zeros(np.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        r = np.sum(np.square(d[:, 0]) + np.square(d[:, 1]))
        return r

    def jac(p, src, dst):
        T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                          [np.sin(p[2]), np.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = np.size(src, 0)
        xt = np.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = np.zeros(np.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        dUdth_R = np.matrix([[-np.sin(p[2]), -np.cos(p[2])],
                                [np.cos(p[2]), -np.sin(p[2])]])
        dUdth = (src * dUdth_R.T).A
        g = np.array([np.sum(2 * d[:, 0]),
                         np.sum(2 * d[:, 1]),
                         np.sum(2 * (d[:, 0] * dUdth[:, 0] + d[:, 1] * dUdth[:, 1]))])
        return g

    def hess(p, src, dst):
        n = np.size(src, 0)
        T = np.matrix([[np.cos(p[2]), -np.sin(p[2]), p[0]],
                          [np.sin(p[2]), np.cos(p[2]), p[1]],
                          [0, 0, 1]])
        n = np.size(src, 0)
        xt = np.ones([n, 3])
        xt[:, :-1] = src
        xt = (xt * T.T).A
        d = np.zeros(np.shape(src))
        d[:, 0] = xt[:, 0] - dst[:, 0]
        d[:, 1] = xt[:, 1] - dst[:, 1]
        dUdth_R = np.matrix([[-np.sin(p[2]), -np.cos(p[2])], [np.cos(p[2]), -np.sin(p[2])]])
        dUdth = (src * dUdth_R.T).A
        H = np.zeros([3, 3])
        H[0, 0] = n * 2
        H[0, 2] = np.sum(2 * dUdth[:, 0])
        H[1, 1] = n * 2
        H[1, 2] = np.sum(2 * dUdth[:, 1])
        H[2, 0] = H[0, 2]
        H[2, 1] = H[1, 2]
        d2Ud2th_R = np.matrix([[-np.cos(p[2]), np.sin(p[2])], [-np.sin(p[2]), -np.cos(p[2])]])
        d2Ud2th = (src * d2Ud2th_R.T).A
        H[2, 2] = np.sum(2 * (
                    np.square(dUdth[:, 0]) + np.square(dUdth[:, 1]) + d[:, 0] * d2Ud2th[:, 0] + d[:, 0] * d2Ud2th[
                                                                                                                :, 0]))
        return H

    t0 = time.time()
    init_pose = (0, 0, 0)
    src = np.array([a.T], copy=True).astype(np.float32)
    dst = np.array([b.T], copy=True).astype(np.float32)
    Tr = np.array([[np.cos(init_pose[2]), -np.sin(init_pose[2]), init_pose[0]],
                      [np.sin(init_pose[2]), np.cos(init_pose[2]), init_pose[1]],
                      [0, 0, 1]])
    print("src", np.shape(src))
    print("Tr[0:2]", np.shape(Tr[0:2]))
    src = cv2.transform(src, Tr[0:2])
    p_opt = np.array(init_pose)
    T_opt = np.array([])
    error_max = sys.maxsize
    first = False
    while not (first and time.time() - t0 > max_time):
        distances, indices = sklearn.neighbors.NearestNeighbors(n_neighbors=1, algorithm='auto', p=3).fit(
            dst[0]).kneighbors(src[0])
        p = scipy.optimize.minimize(res, [0, 0, 0], args=(src[0], dst[0, indices.T][0]), method='Newton-CG', jac=jac,
                                    hess=hess).x
        T = np.array([[np.cos(p[2]), -np.sin(p[2]), p[0]], [np.sin(p[2]), np.cos(p[2]), p[1]]])
        p_opt[:2] = (p_opt[:2] * np.matrix(T[:2, :2]).T).A
        p_opt[0] += p[0]
        p_opt[1] += p[1]
        p_opt[2] += p[2]
        src = cv2.transform(src, T)
        Tr = (np.matrix(np.vstack((T, [0, 0, 1]))) * np.matrix(Tr)).A
        error = res([0, 0, 0], src[0], dst[0, indices.T][0])

        if error < error_max:
            error_max = error
            first = True
            T_opt = Tr

    p_opt[2] = p_opt[2] % (2 * np.pi)

    return T_opt, error_max


def alpha_shape(points: np.ndarray, alpha: float, only_outer=True) -> tuple:
    """
    Compute the alpha shape (concave hull) of a set of points.

    Parameters
    ----------
    points
        np.array of shape (n,2) points.
    alpha
        alpha value.
    only_outer
    boolean value to specify if we keep only the outer border or also inner edges.

    Return
    ----------
    Set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.

    Refer
    ----------
    https://stackoverflow.com/questions/50549128/boundary-enclosing-a-given-set-of-points
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points, if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    circum_r_list = []
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        circum_r_list.append(circum_r)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    boundary = list(set(list(chain.from_iterable(list(edges)))))  # noqa
    return boundary, edges, circum_r_list


def rotate_via_numpy(xy: np.ndarray, radians: float) -> np.ndarray:
    """
    Use numpy to build a rotation matrix and take the dot product.

    Parameters
    ----------
    xy
        coordinate
    radians
        rotation radians

    Return
    ----------
    Rotated coordinate

    Refer
    ----------
    https://gist.github.com/LyleScott/e36e08bfb23b1f87af68c9051f985302
    """
    print(f"Rotation {radians * 180 / np.pi} degree")
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, xy.T).T
    return np.array(m)


def ndt_pre_match(dataLoader1, dataLoader2, grid_resolution_src=0.1, grid_resolution_dist=0.1):
    """
    NDT algorithm for coordinate registration.

    Args:
        dataLoader1: SpatialFuserDataLoader. The first dataset loader containing source coordinates.
        dataLoader2: SpatialFuserDataLoader. The second dataset loader containing target coordinates.
        grid_resolution_src: float. Grid resolution used for the source dataset.
        grid_resolution_dist: float. Grid resolution used for the target dataset.

    Returns:
            The source and target dataset loaders with registered coordinates.
    """

    dst_points = dataLoader1.adata.obsm['spatial']
    src_points = dataLoader2.adata.obsm['spatial']

    src_grids = Grids(src_points, grid_resolution=grid_resolution_src)
    # src_grids.show_score_distribution()

    dst_grids = Grids(dst_points, grid_resolution=grid_resolution_dist)
    # dst_grids.show_score_distribution()

    compute_score_with_grids = partial(compute_grids_score, src_points=src_points, dst_grids=dst_grids)

    initial_guess = [0, 0, 0]
    res = minimize(compute_score_with_grids, initial_guess, method='Powell',
                   options={'maxiter': 1000, 'disp': True})

    trans = res.x
    transformed_points_final = apply_transformation(src_points, trans)

    dataLoader2.adata.obsm['spatial'] = transformed_points_final
    dataLoader2.data.y[:, [2, 3]] = torch.tensor(transformed_points_final)

    # plot_results(src_points, dst_points, transformed_points_final, 'NDT')

    print(f"Estimated transform: {res.x}")
    return dataLoader1, dataLoader2


def icp_pre_match(dataLoader1, dataLoader2, alpha=25, only_outer=True, show_boundary=False):
    """
    ICP algorithm for coordinate registration.

    Args:
        dataLoader1: SpatialFuserDataLoader. The source dataset loader containing the input coordinates.
        dataLoader2: SpatialFuserDataLoader. The target dataset loader containing the reference coordinates.
        alpha: float. Step size or learning rate for the registration update.
        only_outer: bool. If True, restricts registration to outer boundary points only.
        show_boundary: bool. If True, displays the boundary during the registration process.

    Returns:
        The source and target dataset loaders with registered coordinates.
    """

    spatial_info1 = dataLoader1.adata.obsm['spatial']
    spatial_info2 = dataLoader2.adata.obsm['spatial']

    boundary_1, edges_1, _ = alpha_shape(spatial_info1, alpha=alpha, only_outer=only_outer)
    boundary_2, edges_2, _ = alpha_shape(spatial_info2, alpha=alpha, only_outer=only_outer)
    if show_boundary:
        plt.scatter(spatial_info1[:, 0], spatial_info1[:, 1], s = 0.5)
        for i, j in edges_1:
            plt.plot(spatial_info1[[i, j], 0], spatial_info1[[i, j], 1])

        plt.text(270.5,459, f"alpha={alpha}", size=18)
        plt.show()
        plt.scatter(spatial_info2[:, 0], spatial_info2[:, 1], s = 0.5)
        for i, j in edges_2:
            plt.plot(spatial_info2[[i, j], 0], spatial_info2[[i, j], 1])

        plt.text(270.5,459, f"alpha={alpha}", size=18)
        plt.show()

    T, error = icp(spatial_info2[boundary_2,:].T,spatial_info1[boundary_1,:].T)
    dx = T[0,2]
    dy = T[1,2]
    rotation = np.arcsin(T[0,1]) * 360 / 2 / np.pi

    print("T",T)
    print("error",error)
    print("rotation°",rotation)

    trans = np.squeeze(cv2.transform(np.array([spatial_info2], copy=True).astype(np.float32), T))[:,:2]
    dataLoader2.adata.obsm['spatial'] = trans
    plot_results(spatial_info2, spatial_info1, trans, 'ICP')
