"""
A manager to facilitate instance-oriented demonstration NER. 
"""

from src.data import Instance
from typing import List, Dict, Set


class SearchSpaceManager:
    """
        In instance-oriented NER we sometimes have restrictions on the search space.
        This singleton object helps with managing search spaces. 
    """

    def __init__(self, instances: List[Instance]):
        self._insts = instances

        # Extract all possible labels in this instance dataset
        self._labels = set()
        for inst in self._insts:
            for entity, label in inst.entities:
                self._labels.add(label)

        # Assign each possible label an index and build a mapping from label to index
        # This helps building the combination search spaces by giving each label an encoding.
        self._id2lb = list(self._labels)
        self._lb2id = dict((s, i) for i, s in enumerate(self._lb2id))

        self._built_combination_spaces = self._built_single_label_spaces = False

    def build_combination_search_spaces(self):
        """
            Build a search space for each possible combination of labels using bitmask
        """
        # Because there should be 2^n search spaces, n cannot be too large.
        assert len(self._labels) < 10, \
            f"N={len(self._labels)}, too large to build combination search spaces!"

        N = (1 << len(self._labels))
        self._comb_space = [[]] * N
        # Enumerate all the combinations of labels
        for inst in self._insts:
            state = self.__get_mask(inst)
            for msk in range(N):
                # determine if this instance's labels is a subset of this mask
                if (msk | state) == msk:
                    self._comb_space[msk].append(inst)

        self._built_combination_spaces = True

    def build_single_label_search_spaces(self):
        """
            Build a search space for each label, which contains all the instances that appear in this label.
            An instance may appear in multiple single label search spaces.
        """
        N = len(self._labels)
        self._singlelb_space = [[]] * N
        for inst in self._insts:
            for lb in set(label for entity, label in inst.entities):
                self._singlelb_space[self._lb2id[lb]].append(inst)

        self._built_single_label_spaces = True

    def single_label_search_space(self, label: str) -> List[Instance]:
        """
            Get the search space of one single label.
            Return the space that contains all the instances that has this label.
        """
        if not self._built_single_label_spaces:
            self.build_single_label_search_spaces()

        # Sanity Check: manager should have seen the label.
        assert label in self._labels, f"Label {label} not in dataset passed to search space manager!"

        return self._singlelb_space[self._lb2id[label]]

    def superset_labels_search_space(self, inst: Instance) -> List[Instance]:
        """
            Get the search space of sentences whos labels are supersets of this instance.
        """
        if not self._built_combination_spaces:
            self.build_combination_search_spaces()

        # Sanity Check: manager should have seen all the labels.
        for label in set(label for entity, label in inst.entities):
            assert label in self._labels, f"Label {label} not in dataset passed to search space manager!"

        return self._comb_space[self.__get_mask(inst)]

    def __get_mask(self, inst: Instance):
        """
            Get the bitmask state of the labels appeared in an entity.
        """
        mask = 0
        labels = set(label for entity, label in inst.entities)
        for s in labels:
            mask |= (1 << self._lb2id[s])
        return mask
