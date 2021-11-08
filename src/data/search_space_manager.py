#
# @author: Kangmin
#

from src.data import Instance
from typing import List, Dict, Set


class SearchSpaceManager:
    """
        A singleton manager object that helps with search space restriction in instance-oriented demonstration. 
    """

    def __init__(self, instances: List[Instance], transformation_func=None):
        self._insts = instances
        self._transformation_func = transformation_func

        # Extract all possible labels in this instance dataset
        self._labels = set()
        for inst in self._insts:
            for entity, label in inst.entities:
                self._labels.add(label)

        # Assign each possible label an index and build a mapping from label to index
        # This helps building the combination search spaces by giving each label an encoding.
        self._id2lb = list(self._labels)
        self._lb2id = d = dict((s, i) for i, s in enumerate(self._id2lb))

        self._built_combination_spaces = self._built_single_label_spaces = False

    def build_combination_search_spaces(self):
        """
            Build a search space for each possible combination of labels using bitmask. 
            Note that for a combination x, we choose the instances whose label combinations are supersets of x.
            So an instance might appear in multiple search spaces.
        """
        # Because there should be 2^n search spaces, n cannot be too large.
        assert len(self._labels) < 10, \
            f"N={len(self._labels)}, too large to build combination search spaces!"

        N = (1 << len(self._labels))
        self._comb_space = [[] for _ in range(N)]
        # Enumerate all the combinations of labels
        for inst in self._insts:
            state = self.__get_mask(inst)
            for msk in range(N):
                # determine if this instance's labels is a superset of this mask
                if (msk & state) == msk:
                    self._comb_space[msk].append(inst)

        if self._transformation_func is not None:
            for i in range(N):
                self._comb_space[i] = self._transformation_func(self._comb_space[i])

        self._built_combination_spaces = True

    def build_single_label_search_spaces(self):
        """
            Build a search space for each label, which contains all the instances that have this label.
            An instance may appear in multiple single label search spaces.
        """
        N = len(self._labels)
        self._singlelb_space = [[] for _ in range(N)]
        for inst in self._insts:
            for lb in set(label for entity, label in inst.entities):
                self._singlelb_space[self._lb2id[lb]].append(inst)

        if self._transformation_func is not None:
            for i in range(N):
                self._singlelb_space[i] = self._transformation_func(self._singlelb_space[i])

        self._built_single_label_spaces = True

    def single_label_search_space(self, label: str) -> List[Instance]:
        """
            Returns the space that contains all the instances that has this label.
        """
        if not self._built_single_label_spaces:
            self.build_single_label_search_spaces()

        # Sanity Check: manager should have seen the label.
        assert label in self._labels, f"Label {label} not in dataset passed to search space manager!"

        return self._singlelb_space[self._lb2id[label]]

    def superset_labels_search_space(self, inst: Instance) -> List[Instance]:
        """
            Returns the search space of instances whose labels are supersets of instance inst.
        """
        if not self._built_combination_spaces:
            self.build_combination_search_spaces()

        # Sanity Check: manager should have seen all the labels.
        for label in set(label for entity, label in inst.entities):
            assert label in self._labels, f"Label {label} not in dataset passed to search space manager!"

        return self._comb_space[self.__get_mask(inst)]

    def __get_mask(self, inst: Instance):
        """
            Get the bitmask state of an instance based on the labels it has.
        """
        mask = 0
        labels = set(label for entity, label in inst.entities)
        for s in labels:
            mask |= (1 << self._lb2id[s])
        return mask
