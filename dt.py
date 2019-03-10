import numpy as np
import pandas as pd


class Node():

    def __init__(self, x, y, algorithm = None):
        self.algorithm = algorithm


    def split(self, algorithm = None):
        pass


class Tree():

    def __init__(self, x, y, algorithm=None, max_depth = 10, min_elements_in_bin = 1):
        self.algorithm = algorithm
        self.x = x
        self.y = y

    def fit(self):
        pass