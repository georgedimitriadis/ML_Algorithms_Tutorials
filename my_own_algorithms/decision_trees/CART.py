
from typing import Any, Self, Callable
import numpy as np
import pandas as pd
from copy import copy

class Region:
    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.empty = True if len(X) == 0 else False

class Node:
    def __init__(self, id: int, region: Region, partition_fn: Callable, parent_node: type(Self) | None = None,
                 side: bool | None = None, forced_leaf: bool = False) -> None:
        self.id = id
        self.region: Region = region
        self.parent_node: Node = parent_node
        self.side: bool = side
        self.forced_leaf = forced_leaf
        self._n_children: int = 0
        self.at_depth: int = 0 if self.parent_node is None else self.parent_node.at_depth + 1
        self.all_children_nodes: list[Node] = []
        self.immediate_children_nodes: list[Node | None] = [None, None]
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
        :return:
        """
        if parent_node is not None:
            depth = parent_node.at_depth + 1
        else:
            depth = 0

        self.n_node_samples.append(len(region.X))
        node_id = self.nodes[-1].id + 1 if len(self.nodes) > 0 else 0
        # Forced to be leaf if we have reached maximum allowed depth or the node cannot be split any further because
        # it has only one unique X value for multiple y values. More 'ending' criteria can be added here.
        forced_leaf = True if (self.max_depth == depth or np.unique(region.X, axis=0).shape[0] == 1) else False

        self.nodes.append(Node(node_id, region=region, parent_node=parent_node, partition_fn=self.partition_fn,
                               side=side, forced_leaf=forced_leaf))
        self.feature.append(self.nodes[-1].feature)
        self.threshold.append(self.nodes[-1].threshold)
        self.n_nodes = self.n_nodes + 1

        if parent_node is not None:
            if side:
                parent_node.immediate_children_nodes[0] = self.nodes[-1]
            else:
                parent_node.immediate_children_nodes[1] = self.nodes[-1]

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
            if not in_nodes:
                break
            alpha_effs = []
            for in_node in in_nodes:
                leaves = [n for n in in_node.all_children_nodes if n.leaf]
                error_of_leaves = np.sum([l.error for l in leaves])
                error_in_node = in_node.error
                alpha_effs.append((error_in_node - error_of_leaves)/ (len(leaves) - 1))
            alpha_eff = np.min(alpha_effs)
            if self.ccp_alpha * (1 + 1e-6) < alpha_eff:
                break

            alpha_eff_index = np.argmin(alpha_effs)
            node_id = in_nodes[alpha_eff_index].id
            print(f'Collapsing Node id {node_id} with alpha_eff {alpha_eff}')
            self._collapse_node(node_id)

    def _get_nodes_from_ids(self, ids: list[int]) -> list[Node]:
        node_list = []
        for id in ids:
            for n in self.nodes:
                if n.id == id:
                    node_list.append(n)
        return node_list

    def _get_index_from_id(self, id):
        index = None
        for i, n in enumerate(self.nodes):
            if n.id == id:
                index = i
        return index

    def _collapse_node(self, id):
        node = self._get_nodes_from_ids([id])[0]
        node.leaf = True
        children_to_remove = copy(node.immediate_children_nodes)
        if None not in children_to_remove:
            for child in children_to_remove:
                self._remove_node(child)


    def _remove_node(self, node: Node):
        children_nodes = copy(node.immediate_children_nodes)
        if None not in children_nodes:
            for child in children_nodes:
                self._remove_node(child)

        for n in self.nodes:
            if node in n.all_children_nodes:
                n.all_children_nodes.remove(node)
            if node in n.immediate_children_nodes:
                n.immediate_children_nodes.remove(node)

        index = self._get_index_from_id(node.id)
        self.nodes.pop(index)
        self.feature.pop(index)
        self.n_node_samples.pop(index)
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
        max_depth = 0

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

class DecisionTreeRegressor:

    def __init__(self, criterion: str, max_depth: int | None = None, ccp_alpha: float = 0.0):
        if criterion == 'squared_error':
            self.criterion = self._squared_error
        elif criterion == 'absolute_error':
            self.criterion = self._absolute_error
        else:
            print('Unrecognised criterion. Not functional regressor')
            return
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.tree = Tree(partition_fn=self._partition, max_depth=self.max_depth, ccp_alpha=ccp_alpha)

    def fit(self, X, y):
        self.tree.grow(X, y)
        if self.ccp_alpha > 0:
            self.tree.cost_complexity_pruning()

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
            while not node.leaf:
                if x[node.feature] <= node.threshold:
                    node = [n for n in self.tree.nodes if n.parent_node == node and n.side][0]
                else:
                    node = [n for n in self.tree.nodes if n.parent_node == node and not n.side][0]
            if self.criterion == self._squared_error:
                y[i] = np.mean(node.region.y)
            elif self.criterion == self._absolute_error:
                y[i] = np.median(node.region.y)

        return y

    def _squared_error(self, y: np.ndarray | list, y_hat_R1: np.floating[Any], y_hat_R2: np.floating[Any],
                       ind_R1: np.ndarray | list, ind_R2: np.ndarray | list) -> float:
        return (np.sum([np.square(y[i] - y_hat_R1) for i in ind_R1]) + np.sum([np.square(y[i] - y_hat_R2) for i in ind_R2])) / len(y)

    def _absolute_error(self, y: np.ndarray | list, y_hat_R1: np.floating[Any], y_hat_R2: np.floating[Any],
                        ind_R1: np.ndarray | list, ind_R2: np.ndarray | list) -> float:
        return (np.sum([np.abs(y[i] - y_hat_R1) for i in ind_R1]) + np.sum(np.abs([y[i] - y_hat_R2 for i in ind_R2]))) / len(y)

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
        for j in range(n_features):
            x_j = np.unique(np.sort(X[:, j]))
            if len(x_j) == 1:
                continue
                #ss = x_j
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

            if error_min <= error_j_min:
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
        error = error / self.tree.n_node_samples[0]
        return R1, R2, feature, threshold, error


class DecisionTreeClassifier:

    def __init__(self, criterion: str, max_depth: int | None = None, ccp_alpha: float = 0.0):
        if criterion == 'gini':
            self.criterion = self._gini
        elif criterion == 'entropy':
            self.criterion = self._entropy
        else:
            print('Unrecognised criterion. Not functional classifier')
            return
        self.max_depth = max_depth
        self.ccp_alpha = ccp_alpha
        self.tree = Tree(partition_fn=self._partition, max_depth=self.max_depth, ccp_alpha=ccp_alpha)

    def fit(self, X, y):
        self.tree.grow(X, y)
        if self.ccp_alpha > 0:
            self.tree.cost_complexity_pruning()

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
            while not node.leaf:
                if x[node.feature] <= node.threshold:
                    node = [n for n in self.tree.nodes if n.parent_node == node and n.side][0]
                else:
                    node = [n for n in self.tree.nodes if n.parent_node == node and not n.side][0]
            unique_values = np.unique(node.region.y)
            if len(unique_values) == 1:
                y[i] = int(node.region.y[0])
            else:
                # If the leaf is not pure then get the class with the highest probability. If there is a tie break it with the earliest class
                probs = []
                for v in unique_values:
                    probs.append(np.sum(node.region.y == v) / len(node.region.y))
                sorted_probs, sorted_values = map(list, zip(*sorted(zip(probs, unique_values))))
                equal_probs_values = []
                for k, p in enumerate(sorted_probs[:-1]):
                    if p == sorted_probs[-1]:
                        equal_probs_values.append(k)
                if len(equal_probs_values) >0:
                    y[i] = sorted_values[equal_probs_values[0]]
                else:
                    y[i] = sorted_values[-1]

        return y

    def _gini(self, class_proba: list) -> float:
        class_proba = np.array(class_proba)
        return np.sum(class_proba *(1 - class_proba))

    def _entropy(self, class_proba: list) -> float:
        return -np.sum(class_proba * np.log(class_proba))

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
        for j in range(n_features):
            x_j = np.unique(np.sort(X[:, j]))
            if len(x_j) == 1:
                continue
                #ss = x_j
            else:
                ss = [(x_j[k] + x_j[k + 1]) / 2 for k in range(len(x_j) - 1)]
            ss_min = ss[0]
            error_min = np.inf
            for s in ss:
                ind_R1, ind_R2 = calc_indices(j, s)
                if len(ind_R1) > 0:
                    class_proba_R1 = 1/len(y[ind_R1]) * np.array([np.sum(y[ind_R1]==i) for i in np.unique(y[ind_R1])])
                    impurity_R1 = self.criterion(class_proba_R1)
                else:
                    impurity_R1 = 0
                if len(ind_R2) > 0:
                    class_proba_R2 = 1 / len(y[ind_R2]) * np.array([np.sum(y[ind_R2] == i) for i in np.unique(y[ind_R2])])
                    impurity_R2 = self.criterion(class_proba_R2)
                else:
                    impurity_R2 = 0
                error = (len(ind_R1) / len(y)) * impurity_R1 + (len(ind_R2) / len(y) * impurity_R2)

                if error < error_min:
                    error_min = error
                    ss_min = s

            if error_min <= error_j_min:
                j_min = j
                s_min = ss_min
                error_j_min = error_min

        ind_R1, ind_R2 = calc_indices(j_min, s_min)
        R1 = Region(X[ind_R1, :], y[ind_R1])
        R2 = Region(X[ind_R2, :], y[ind_R2])

        feature = int(j_min)
        threshold = float(s_min)
        probs = 1/len(y) * np.array([np.sum(y==i) for i in np.unique(y)])
        error = len(y) / self.tree.n_node_samples[0] * self.criterion(probs)

        return R1, R2, feature, threshold, error


# Regression
reg_test_set = pd.read_csv('data/decission_tree_regression_test.csv')
X_test = reg_test_set[['x1', 'x2']].to_numpy()
y_test = reg_test_set['y_expected'].to_numpy()
y_test_depth4 = reg_test_set['y_maxdepth4'].to_numpy()
y_test_alpha = reg_test_set['y_ccp_alpha_0.6'].to_numpy()
X_train = np.array([
    [5, 1], [7, 1], [6, 3], [7, 4], [7, 5],
    [6, 1], [4, 5], [8, 3], [5, 5], [7, 2],
], dtype=float)
y_train = np.array([15.0, 12.0, 19.0, 21.0, 21.0, 12.0, 8.0, 18.0, 21.0, 13.0])
names = ["x1", "x2"]

reg = DecisionTreeRegressor('squared_error', max_depth = None)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
np.testing.assert_almost_equal(y_hat, y_test, decimal=3)

reg = DecisionTreeRegressor('squared_error', max_depth = 3)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
np.testing.assert_almost_equal(y_hat, y_test_depth4, decimal=3)

reg = DecisionTreeRegressor('squared_error', max_depth = None, ccp_alpha=0.6)
reg.fit(X_train, y_train)
y_hat = reg.predict(X_test)
np.testing.assert_almost_equal(y_hat, y_test_alpha, decimal=3)


# Classification
X_train=np.array([[4,6,1],[5,4,0],[8,4,0],[7,1,1],[6,4,0],[5,2,1],[8,2,0],[5,6,1],
            [1,3,0],[5,3,0],[7,2,1],[7,5,0],[7,5,0]],dtype=float)
y_train=np.array([1,0,0,2,0,2,1,2,0,1,1,2,1]); classes=["A","B","C"]; names=["x1","x2","x3"]


cls_test_set = pd.read_csv(r'E:\Code\Mine\Learning_ML_RL\ML_Algorithms_Tutorials\data\decission_tree_classification_test.csv')
X_test = cls_test_set[['x1', 'x2', 'x3']].to_numpy()
y_test = cls_test_set['class_expected'].to_numpy()
y_test_depth4 = cls_test_set['class_maxdepth4'].to_numpy()
y_test_alpha = cls_test_set['class_ccp_alpha_0.08'].to_numpy()

letters = ['A', 'B', 'C']


cls = DecisionTreeClassifier(criterion='gini')
cls.fit(X_train, y_train)
y_hat_nums = cls.predict(X_test)
y_hat = [letters[int(i)] for i in y_hat_nums]
print(np.all(y_hat == y_test))

cls = DecisionTreeClassifier(criterion='gini', max_depth = 3)
cls.fit(X_train, y_train)
y_hat_nums = cls.predict(X_test)
y_hat = [letters[int(i)] for i in y_hat_nums]
print(np.all(y_hat == y_test_depth4))


cls = DecisionTreeClassifier(criterion='gini', ccp_alpha=0.08)
cls.fit(X_train, y_train)
y_hat_nums = cls.predict(X_test)
y_hat = [letters[int(i)] for i in y_hat_nums]
print(np.all(y_hat == y_test_alpha))