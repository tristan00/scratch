import numpy as np
import pandas as pd
from graphviz import Digraph
import uuid
from sklearn.metrics import mean_squared_error
import random
import functools
import operator


class Node():

    def __init__(self, data, max_depth, min_num_in_bin = 1, algorithm = None, target = 'target', depth = 0):
        self.data = data
        self.algorithm = algorithm
        self.depth = depth
        self.target = target
        self.split_feature = None
        self.split_value = None
        self.child_nodes = []
        self.max_depth = max_depth
        self.min_num_in_bin = min_num_in_bin
        self.id = str(uuid.uuid4())
        self.max_splits_to_consider = 10
        self.mean_value = self.data[self.target].mean()
        self.error_value = None


    def split(self):
        best_gain = np.inf
        best_feature = None
        best_split_value = None


        if self.min_num_in_bin > self.data.shape[0] or np.std(self.data[self.target].values) == 0.0 or self.depth >= self.max_depth:
            return


        for i in self.data.columns:
            if i != self.target:
                split_v = self.data[i].mean()
                std_1 = np.std(self.data[self.data[i] < split_v][self.target].values)
                std_2 = np.std(self.data[self.data[i] >= split_v][self.target].values)

                if std_1 + std_2 < best_gain:
                    best_gain = std_1 + std_2
                    best_feature = i
                    best_split_value = split_v

        if best_feature:
            self.split_feature = best_feature
            self.split_value = best_split_value

            self.child_nodes = []
            self.child_nodes.append(Node(self.data[self.data[self.split_feature] < self.split_value], self.max_depth,
                                         target = self.target, depth = self.depth + 1, algorithm=self.algorithm))
            self.child_nodes.append(Node(self.data[self.data[self.split_feature] >= self.split_value], self.max_depth,
                                         target = self.target, depth = self.depth + 1, algorithm=self.algorithm))
            for i in self.child_nodes:
                i.split()


    def predict(self, data_point):
        if self.child_nodes and self.algorithm in ['id3_reg', 'random_greedy']:
            if data_point[self.split_feature] < self.split_value:
                return self.child_nodes[0].predict(data_point)
            else:
                return self.child_nodes[1].predict(data_point)

        return self.mean_value

    def get_all_children(self):
        return functools.reduce(operator.add, [i.get_all_children() for i in self.child_nodes], [])

    def return_self_dict(self):
        return {'n_id': self.id,
                   'data_size':self.data.shape[0],
                   'split_feature':self.split_feature,
                   'split_value':self.split_value}


class Tree():

    def __init__(self, data, algorithm = 'id3_reg', max_depth = 10, min_elements_in_bin = 1):
        self.algorithm = algorithm
        self.data = data
        self.root_node = Node(data, max_depth, algorithm=algorithm)

        self.nodes = dict()

        self.nodes[self.root_node.id] = self.root_node


    def fit(self):
        self.root_node.split()

        for i in self.root_node.get_all_children():
            self.nodes[i.id] = i


    def predict(self, data):
        results = []

        if len(data.shape) == 2:
            for k, v in data.iterrows():
                results.append(self.root_node.predict(v))
            return np.array(results)
        elif len(data.shape) == 1:
            return self.root_node.predict(data)

        raise Exception('Invalid data shape: {0}'.format(data.shape))


    def visualize(self):
        tree = Digraph()

        nodes = [self.root_node]
        while nodes:
            next_nodes = []

            for node in nodes:

                node_desc = 'Split_feature: {0},  split_value: {1}, avg_target_value: {2}'.format(node.split_feature, node.split_value, node.data[node.target].mean())

                tree.node(node.id, depth = str(node.depth), label = node_desc)

                if len(node.child_nodes) == 2:
                    tree.edge(node.id, node.child_nodes[0].id, label = '<')
                    tree.edge(node.id, node.child_nodes[1].id, label = '>=')

                else:
                    for child_node in node.child_nodes:
                        tree.edge(node.id, child_node.id)

                next_nodes.extend(node.child_nodes)

            nodes = next_nodes
        return tree.source

    def get_split_info(self):
        splits = []


        rn_info = self.root_node.return_self_dict()
        splits.append(rn_info)





if __name__ == '__main__':
    from sklearn.tree import DecisionTreeRegressor, export_graphviz
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_absolute_error
    import time

    data = fetch_california_housing()
    data_df = pd.DataFrame(data = np.hstack([data.data, np.expand_dims(data.target, 1)]),
                           columns = data.feature_names + ['target'])

    train_df, val_df = train_test_split(data_df, train_size=.25, random_state=1)
    start_time = time.time()
    t1 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=2)
    t1.fit(train_df.drop('target', axis=1), train_df['target'])
    print('t1 run time', time.time() - start_time)

    start_time = time.time()
    t2 = Tree(train_df, max_depth=4)
    t2.fit()
    print('t2 run time', time.time() - start_time)

    t1_preds = t1.predict(val_df.drop('target', axis=1))
    t2_preds = t2.predict(val_df.drop('target', axis=1))

    print('t1 preds: {0}'.format(r2_score(val_df['target'], t1_preds)))
    print('t2 preds: {0}'.format(r2_score(val_df['target'], t2_preds)))

    # print(t2.visualize())
    #
    # print(export_graphviz(t1))


