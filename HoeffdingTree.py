# Author: Niek Hasselerharm (s1026769)
# Created as part of a project for the course NWI-IBI008 Data Mining.
#
# An implementation of the VFDT described in the paper "Mining High-Speed Data Streams" by Domingos & Hulten.
# See: http://web.cs.wpi.edu/~cs525/f13b-EAR/cs525-homepage/lectures/PAPERS/p71-domingos.pdf


# Project-file imports
from HTNode import HTNode


# The HoeffdingTree class is an implementation of the VFDT algorithm, based on the Hoeffding Tree algorithm.
class HoeffdingTree:
    def __init__(self, n_min, d, t, attributes):
        self.n_min = n_min              # Minimum number of samples a node has to have processed before
        self.d = d                      # The delta used in the computation of the Hoeffding bound.
        self.t = t                      # The user-specified tau threshold used in the ([difference in G] < epsilon < tau) check in case of a "tie".
        self.root = HTNode(attributes)  # The root node of the tree.

    # Trains the tree by inserting a new sample and updating the tree.
    def train(self, x, y):
        # Find leaf-node that sample x belongs to and update that node's statistics.
        leaf = self.root.find(x)
        leaf.update(x, y)

        # Evaluate splitting of leaf node after addition of new sample.
        split = leaf.evaluate_split(self.n_min, self.d, self.t)
        if split is not None:
            split_attr, split_val = split
            leaf.split(split_attr, split_val)

    # Predicts the class of sample x by using the tree.
    def predict(self, x):
        leaf = self.root.find(x)
        return leaf.get_class()
