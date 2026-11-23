
from typing import Any, Self, Callable
import numpy as np
from copy import copy

from torch.utils import _get_clean_triton

# Regression test data set
X = np.array([
    [5, 1], [7, 1], [6, 3], [7, 4], [7, 5],
    [6, 1], [4, 5], [8, 3], [5, 5], [7, 2],
], dtype=float)
y = np.array([15.0, 12.0, 19.0, 21.0, 21.0, 12.0, 8.0, 18.0, 21.0, 13.0])
names = ["x1", "x2"]
# ------------------------------------------

class Region:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.empty = True if len(X) == 0 else False

class Node:
    def __init__(self, id: int, region: Region, partition_fn: Callable, parent_node: Self | None = None, side: bool | None = None, forced_leaf: bool = False) -> None:
        self.id = id
        self.region: Region = region
        self.parent_node: Node = parent_node
        self.side: bool = side
        self.forced_leaf = forced_leaf
        self._n_children: int = 0
        self.at_depth: int = 0 if self.parent_node is None else self.parent_node.at_depth + 1
        self.all_children_nodes: list[Node] = []
        self.immediate_children_nodes: list[Node] = [None, None]
        self._partition: Callable = partition_fn

        if self.parent_node is not None:
            self.parent_node.n_children += 1
            p = self.parent_node
            while p is not None:
                p.all_children_nodes.append(self)
                p = p.parent_node

        self.pure: bool = self._check_if_pure()
        self.leaf: bool = self._check_if_leaf()

        if self.leaf:
            self.l_r_completed: list[bool] = [True, True]
            self.feature: int = -1
            self.threshold: float = -1
            self.region_left: Region = Region([], [])
            self.region_right: Region = Region([], [])
            _, _, _, _, self.error = self._partition(self.region)
        else:
            self.l_r_completed: list[bool] = [False, False]
            self.region_left, self.region_right, self.feature, self.threshold, self.error = self._partition(self.region)

        self.impurity = self.error / len(self.region.y)

    @property
    def n_children(self):
        return self._n_children

    @n_children.setter
    def n_children(self, n_ch):
        self._n_children = n_ch
        if self.parent_node is not None:
            self.parent_node.n_children += 1

    def _check_if_pure(self):
        # if all y values are the same
        if len(np.unique(self.region.y)) == 1:
            return True
        return False

    def _check_if_leaf(self):
        if self.pure or len(np.unique(self.region.X)) == 1 or self.forced_leaf:
            return True
        return False

class Tree:
    def __init__(self, partition_fn, ccp_alpha: float = 0.0, max_depth: int | None = None) -> None:
        """
        The Tree class defines a tree structure of Nodes.
        :param partition_fn: This is the partition function that each Node will use to partition its region.
            This is tree type dependent (Regression vs Classification) and is implemented in the specific DecisionTree class.
        :param ccp_alpha: The alpha for cost-complexity pruning. Default 0.0
        :param max_depth: The maximum depth the tree is allowed to grow. Default None
        """
        self._initialise_tree()
        self.partition_fn = partition_fn
        self.ccp_alpha = ccp_alpha
        self.max_depth = max_depth

    def _initialise_tree(self):
        self.nodes = []
        self.children_left = []
        self.children_right = []
        self.feature = []
        self.threshold = []
        self.n_node_samples = []
        self.impurity = []
        self.weighted_n_node_samples = []
        self.value = []
        self.depth = 0
        self.n_nodes = 0

    def add_node(self, region: Region, parent_node: Node | None, side: bool | None):
        """
        Add a new Node to the tree
        :param region: The X, y region of the Node
        :param parent_node: The parent Node
        :param side: Is this the left (True) or Right (False) side of the parent Node
        :parma ids_to_force_leaf: If the id of this Node is in the ids_to_force_leaf array then force it to be a leaf.
        :return:
        """

        if ids_to_force_leaf is None:
            ids_to_force_leaf = []

        self.n_node_samples.append(len(region.X))
        node_id = self.nodes[-1].id + 1 if len(self.nodes) > 0 else 0
        if parent_node is not None:
            depth = parent_node.at_depth + 1
            if side:
                parent_node.immediate_children_nodes[0] = self.nodes[-1]
            else:
                parent_node.immediate_children_nodes[-1] = self.nodes[-1]
        else:
            depth = 0
        forced_leaf = True if (self.max_depth == depth) else False
        self.nodes.append(Node(node_id, region=region, parent_node=parent_node, partition_fn=self.partition_fn,
                               side=side, forced_leaf=forced_leaf))
        self.feature.append(self.nodes[-1].feature)
        self.threshold.append(self.nodes[-1].threshold)
        self.impurity.append(self.nodes[-1].impurity)
        self.n_nodes = self.n_nodes + 1

    def _calculate_num_of_l_r_children_per_node(self):
        self.children_left = np.zeros((self.n_nodes))
        self.children_right = np.zeros((self.n_nodes))

        for i, node in enumerate(self.nodes):
            if len(node.all_children_nodes) > 0:
                for child in node.all_children_nodes:
                    if child.parent_node == node and child.side:
                        self.children_left[i] = len(child.all_children_nodes) + 1
                    if child.parent_node == node and not child.side:
                        self.children_right[i] = len(child.all_children_nodes) + 1
            else:
                self.children_left[i] = 0
                self.children_right[i] = 0

    def cost_complexity_pruning(self):
        while True:
            in_nodes = [node for node in self.nodes if not node.leaf]
            alpha_effs = []
            for in_node in in_nodes:
                leaves = [n for n in in_node.all_children_nodes if n.leaf]
                error_of_leaves = np.sum([l.error for l in leaves])
                error_in_node = in_node.error
                alpha_effs.append((error_in_node - error_of_leaves)/(self.n_node_samples[0] * (len(leaves) - 1)))
            alpha_eff = np.min(alpha_effs)
            if self.ccp_alpha * (1-1e-6) < alpha_eff < self.ccp_alpha * (1+1e-6) or self.ccp_alpha < alpha_eff:
                break

            alpha_eff_index = np.argmin(alpha_effs)
            node_id = in_nodes[alpha_eff_index].id
            print(f'Collapsing Node id {node_id} with alpha_eff {alpha_eff}')
            self._collapse_node(node_id)


    def _get_index_from_id(self, id):
        index = None
        for i, n in enumerate(self.nodes):
            if n.id == id:
                index = i

        return index

    def _collapse_node(self, id):
        index = self._get_index_from_id(id)
        node = self.nodes[index]
        node.leaf = True
        children_to_remove = node.all_children_nodes
        for child in children_to_remove:
            self._remove_node(child.id)


    def _remove_node(self, id):
        index = self._get_index_from_id(id)
        children_nodes = self.nodes[index].all_children_nodes
        if len(children_nodes) > 0:
            for child in children_nodes:
                ch_ind = self._get_index_from_id(child.id)
                self.nodes.pop(ch_ind)
                self.feature.pop(ch_ind)
                self.n_node_samples.pop(ch_ind)
                self.impurity.pop(ch_ind)
                self.threshold.pop(ch_ind)
                self.children_left = np.delete(self.children_left, ch_ind)
                self.children_right = np.delete(self.children_right, ch_ind)
        index = self._get_index_from_id(id)
        self.nodes.pop(index)
        self.feature.pop(index)
        self.n_node_samples.pop(index)
        self.impurity.pop(index)
        self.threshold.pop(index)
        self.children_left = np.delete(self.children_left, index)
        self.children_right = np.delete(self.children_right, index)

        self.n_nodes = len(self.nodes)

        max_depth = 0
        for n in self.nodes:
            if n.at_depth > max_depth:
                max_depth = n.at_depth

        self.depth = max_depth


    def _re_id_tree(self):
        id = 0
        for d in range(self.depth + 1):
            nodes_at_depth = [n for n in self.nodes if n.at_depth == d]
            for n in nodes_at_depth:
                n.id = id
                id = id + 1


    def grow(self, X, y):
        """
        The main loop that grows the tree according to the criterion chosen
        :param X: The X data array
        :param y: The y data array
        :return:
        """
        self._initialise_tree()
        current_depth = 0
        max_depth = 1

        self.add_node(region=Region(X, y), parent_node=None, side=None)
        root_node = self.nodes[0]
        current_node = root_node

        def move_up(current_node, current_depth, max_depth):
            current_depth = current_depth - 1
            current_node = current_node.parent_node
            if current_depth > max_depth:
                max_depth = current_depth
            return current_node, current_depth, max_depth

        def move_down(current_depth, max_depth):
            current_depth = current_depth + 1
            current_node = self.nodes[-1]
            if current_depth > max_depth:
                max_depth = current_depth
            return current_node, current_depth, max_depth

        while not np.all(root_node.l_r_completed):
            if current_node.pure or current_node.leaf or np.all(current_node.l_r_completed): # End of branch. Go up
                current_node.l_r_completed = [True, True]
                coming_from = current_node.side
                current_node, current_depth, max_depth = move_up(current_node, current_depth, max_depth)

                if coming_from: # Child Node completed was the left side
                    current_node.l_r_completed[0] = True
                else: # Child Node just completed was the right side (so we are done with this Node too)
                    current_node.l_r_completed[1] = True

            elif current_node.l_r_completed[0] and not current_node.l_r_completed[1]: # Have already done the left side. Do the right
                region = current_node.region_right
                self.add_node(region=region, parent_node=current_node, side=False)
                current_node, current_depth, max_depth = move_down(current_depth, max_depth)

            elif not current_node.l_r_completed[0]: # Do the left side first
                region = current_node.region_left
                self.add_node(region=region, parent_node=current_node, side=True)
                current_node, current_depth, max_depth = move_down(current_depth, max_depth)

        self.depth = max_depth
        self._calculate_num_of_l_r_children_per_node()
        self._re_id_tree()

class DecisionTreeRegresor:

    def __init__(self, criterion: str, max_depth: int | None = None, ccp_alpha: float = 0.0):
        if criterion == 'squared_error':
            self.criterion = self._squared_error
        elif criterion == 'absolute_error':
            self.criterion = self._absolute_error
        else:
            print('Unrecognised criterion. Not functional regressor')
            return
        self.max_depth = max_depth
        self.tree = Tree(partition_fn=self._partition, max_depth=self.max_depth, ccp_alpha=ccp_alpha)

    def apply(self, X):
        pass

    def fit(self, X, y):
        self.tree.grow(X, y)

    def get_depth(self):
        return self.tree.depth

    def get_n_leaves(self):
        n_leaves = 0
        for node in self.tree.nodes:
            if node.leaf:
                n_leaves = n_leaves + 1
        return n_leaves

    def predict(self, X):
        y = np.zeros((len(X)))
        for i, x in enumerate(X):
            node = self.tree.nodes[0]
            while True:
                if node.leaf:
                    if self.criterion == self._squared_error:
                        y[i] = np.mean(node.region.y)
                    if self.criterion == self._absolute_error:
                        y[i] = np.median(node.region.y)
                    break
                else:
                    f = node.feature
                    if x[f] < node.threshold:
                        node = node.immediate_children_nodes[0]
                    else:
                        node = node.immediate_children_nodes[1]
        return y

    def _squared_error(self, y: np.ndarray | list, y_hat_R1: np.floating[Any], y_hat_R2: np.floating[Any],
                       ind_R1: np.ndarray | list, ind_R2: np.ndarray | list) -> float:
        return np.sum([np.square(y[i] - y_hat_R1) for i in ind_R1]) + np.sum([np.square(y[i] - y_hat_R2) for i in ind_R2]) / len(y)

    def _absolute_error(self, y: np.ndarray | list, y_hat_R1: np.floating[Any], y_hat_R2: np.floating[Any],
                        ind_R1: np.ndarray | list, ind_R2: np.ndarray | list) -> float:
        return np.sum([np.abs(y[i] - y_hat_R1) for i in ind_R1]) + np.sum(np.abs([np.square(y[i] - y_hat_R2) for i in ind_R2])) / len(y)

    def _partition(self, region: Region) -> tuple[Region, Region, int, float, float]:

        def calc_indices(j, s):
            ind_R1 = np.squeeze(np.argwhere(X[:, j] < s))
            ind_R1 = [int(ind_R1)] if len(ind_R1.shape) == 0 else ind_R1
            ind_R2 = np.squeeze(np.argwhere(X[:, j] >= s))
            ind_R2 = [int(ind_R2)] if len(ind_R2.shape) == 0 else ind_R2

            return ind_R1, ind_R2

        X = region.X
        y = region.y
        n_features = X.shape[1]

        j_min = 0
        s_min = np.inf
        error_j_min = np.inf
        features_impurities = []
        for j in range(n_features):
            x_j = np.unique(np.sort(X[:, j]))
            if len(x_j) == 1:
                ss = x_j
            else:
                ss = [(x_j[k] + x_j[k + 1]) / 2 for k in range(len(x_j) - 1)]
            ss_min = ss[0]
            error_min = np.inf
            for s in ss:
                ind_R1, ind_R2 = calc_indices(j, s)
                if self.criterion == self._squared_error:
                    y_hat_R1 = np.mean(y[ind_R1]) if len(y[ind_R1]) > 1 else y[ind_R1]
                    y_hat_R2 = np.mean(y[ind_R2]) if len(y[ind_R2]) > 1 else y[ind_R2]
                elif self.criterion == self._absolute_error:
                    y_hat_R1 = np.median(y[ind_R1]) if len(y[ind_R1]) > 1 else y[ind_R1]
                    y_hat_R2 = np.median(y[ind_R2]) if len(y[ind_R2]) > 1 else y[ind_R2]
                error = self.criterion(y, y_hat_R1, y_hat_R2, ind_R1, ind_R2)
                if error < error_min:
                    error_min = error
                    ss_min = s

            if error_min < error_j_min:
                j_min = j
                s_min = ss_min
                error_j_min = error_min

        ind_R1, ind_R2 = calc_indices(j_min, s_min)
        R1 = Region(X[ind_R1, :], y[ind_R1])
        R2 = Region(X[ind_R2, :], y[ind_R2])

        feature = int(j_min)
        threshold = float(s_min)
        error = 0
        if self.criterion == self._squared_error:
            error  = np.sum(np.square(y - np.mean(y)))
        elif self.criterion == self._absolute_error:
            error = np.sum(np.abs(y - np.median(y)))

        return R1, R2, feature, threshold, error

reg = DecisionTreeRegresor('squared_error', max_depth = None,ccp_alpha=0.07)
reg.fit(X, y)
print(reg.predict([[5,2], [7.2, 3]]))
self = copy(reg.tree)
self.cost_complexity_pruning()

# Classification test dataset
#X=np.array([[7,6],[3,4],[5,4],[1,1],[8,6],[8,2],[3,3],[5,5],[5,3],[7,4],[1,4],[5,1]],dtype=float)
#y=np.array([2,2,2,2,2,1,0,2,0,2,0,1]); classes=["A","B","C"]; names=["x1","x2"]