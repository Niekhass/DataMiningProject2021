# Author: Niek Hasselerharm (s1026769)
# Created as part of a project for the course NWI-IBI008 Data Mining.
#
# An implementation of the VFDT described in the paper "Mining High-Speed Data Streams" by Domingos & Hulten.
# See: http://web.cs.wpi.edu/~cs525/f13b-EAR/cs525-homepage/lectures/PAPERS/p71-domingos.pdf


# Other library imports
import numpy as np


# The HTNode class is an implementation of a VDFT node. This class performs the updating and altering of the tree and its statistics.
class HTNode:
    def __init__(self, attributes, samples=0):
        self.parent = None                                              # Current node's parent
        self.left = None                                                # Current node's left child
        self.right = None                                               # Current node's right child
        self.split_attribute = None                                     # The attribute that was used to split this node.
        self.split_value = None                                         # The value that was used to split this node.
        self.attributes = attributes                                    # The list of attributes that can be split on.
        self.nijk = {attr: {} for attr in attributes}                   # The dictionary of necessary statistics.
        self.classes = {}                                               # Dictionary of classes with their tally.
        self.samples = samples                                          # Total number of samples processed by the tree.
        self.counter = 0                                                # Counter of samples processed by node.

    # Updates nijk, classes and number of samples after insertion.
    def update(self, x, y):
        # Update general statistics
        self.classes[y] = self.classes.get(y, 0) + 1    # add to or increment entry in classes list.
        self.counter += 1                               # increment counter for minimum samples needed before splitting.
        self.samples += 1                               # increment total number of samples.

        # Update nijk.
        for attr in self.attributes:
            v = x[attr]
            if v not in self.nijk[attr]:                                    # If x's value is not already present in [attr]'s list of values, then
                self.nijk[attr][v] = {y: 1}                                 # add it and add x's class in the value's dictionary of classes.
            else:                                                           # Otherwise, if the value is present in [attr]'s list of values, then
                self.nijk[attr][v][y] = self.nijk[attr][v].get(y, 0) + 1    # add or increment x's class in the value's dictionary of classes.

    # Returns the most common and therefore most likely class of this node.
    # In case this is a newly created leaf-node, its parent's class-list will be used.
    def get_class(self):
        # If this leaf has had no samples yet, then predict using parent's statistics.
        if len(self.classes) == 0:
            return max(self.parent.classes, key=self.parent.classes.get)

        return max(self.classes, key=self.classes.get)

    # Finds the leaf a sample belongs to in the tree.
    def find(self, x):
        # If current node is a leaf, then return self.
        if self.left is None and self.right is None:
            return self

        # Otherwise, recursively continue searching in child nodes.
        if x[self.split_attribute] <= self.split_value:
            return self.left.find(x)
        else:
            return self.right.find(x)

    # Computes the Hoeffding bound (epsilon) based on the Hoeffding inequality.
    def compute_hoeffding(self, d):
        # R is defined as the log of the number of classes.
        r = np.log(len(self.classes))

        # e = sqrt((R^2 * ln(1/d)) / (2 * n))
        e = np.sqrt((r ** 2 * np.log(1 / d)) / (2 * self.samples))
        return e

    # Computes gini impurity of a continuous attribute.
    def continuous_gini(self, values):
        left_classes = {}   # The left partition of classes. (samples with value less than or equal to split value)
        right_classes = {}  # The right partition of classes. (samples with value greater than split value)
        p_left = 0          # The sum of (p_i)^2 for the left partition.
        p_right = 0         # The sum of (p_i)^2 for the right partition.
        min_g = 1           # The current minimum gini impurity found.
        g_val = None        # The split value used to find the current miminum gini impurity.

        # Sort list of values.
        sorted_vals = np.array(sorted(list(values.keys())))

        # Calculate middle points.
        split_vals = (sorted_vals[1:] + sorted_vals[:-1]) / 2

        # Go through all middle points and calculate their gini impurity.
        for splittable_val in split_vals:

            # Partition classes based on sample values.
            for sorted_val in sorted_vals:
                if sorted_val <= splittable_val:
                    side = left_classes
                else:
                    side = right_classes

                # Merge the class counts for all values less than the split value.
                for k in values[sorted_val]:
                    if k in side:
                        side[k] += sorted_val
                    else:
                        side.update({k: sorted_val})

            # Compute left and right gini impurity.
            n_left = sum(left_classes.values())
            n_right = sum(right_classes.values())

            # Check to prevent dividing by zero.
            if not n_left == 0 or n_right == 0:
                for cls in list(left_classes.values()):
                    p_left += (cls / n_left) ** 2

                for cls in list(right_classes.values()):
                    p_right += (cls / n_right) ** 2

            g_left = 1 - p_left
            g_right = 1 - p_right

            # Compute weighted gini impurity and update minimum found gini + value.
            g = (n_left / self.samples) * g_left + (n_right / self.samples) * g_right
            if g < min_g:
                min_g = g
                g_val = splittable_val

        return min_g, g_val

    # Computes gini impurity of current node.
    def null_gini(self):
        p_total = 0  # The sum of (p_i)^2.
        for cls in list(self.classes.keys()):
            p_total += (self.classes[cls] / self.samples) ** 2

        return 1 - p_total

    # Evaluates whether or not current node should split.
    def evaluate_split(self, n_min, d, t):
        # If there is only one class, do not split.
        if len(self.classes) == 1:
            return None

        # If the minimum number of samples has not been passed, do not split.
        if self.counter < n_min:
            return None

        # If there are no remaining attributes to split on, do not split.
        if len(self.attributes) == 0:
            return None

        # Otherwise, start evaluation.
        split_attr = None   # The attribute that has so far produced the minimum gini impurity.
        split_val = None    # The value that has so far produced the minimum gini impurity.
        min_g = 1           # The minimum gini impurity found so far.
        snd_min_g = 1       # The second minimum gini impurity found so far.

        # Determine minimum gini impurity.
        for attr in self.attributes:
            # If current attribute is the attribute this node's parent split on, skip it in this node's consideration.
            if self.parent is not None and attr == self.parent.split_attribute:
                continue

            # Obtain the list of values for current attribute in the nijk dictionary
            values = self.nijk[attr]

            # Continuous attribute gini computation.
            g, g_val = self.continuous_gini(values)

            # Update minimum and 2nd minimum gini if last found gini is new minimum.
            if g < min_g:
                snd_min_g = min_g

                min_g = g
                split_attr = attr
                split_val = g_val

            # Update 2nd minimum if last found gini is new second minimum.
            elif g < snd_min_g:
                snd_min_g = min_g

        # Compute epsilon and gini impurity of null attribute (= gini impurity of current node).
        e = self.compute_hoeffding(d)
        g_null = self.null_gini()

        # Check splitting conditions (G(X_a) - G(X_b) > e and X_a < X_null)
        if snd_min_g - min_g > e and min_g < g_null:
            return split_attr, split_val

        # Check for ties ([difference in g] < e < t) using threshold t:
        if (snd_min_g - min_g) < e < t:
            return split_attr, split_val

        # Otherwise, do not split.
        return None

    # Splits current node by creating left and right child nodes, updating own variables and clearing the dictionary of statistics.
    def split(self, split_attr, split_val):
        self.left = HTNode(self.attributes, self.samples)
        self.right = HTNode(self.attributes, self.samples)
        self.left.parent = self
        self.right.parent = self
        self.split_attribute = split_attr
        self.split_value = split_val
        self.nijk.clear()
