"""
A manager to facilitate instance-oriented demonstration NER. 
"""

from src.data import Instance
from typing import List, Dict


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
        # This will later help building the search spaces
        self._id2lb = list(self._labels)
        self._lb2id = dict((s, i) for i, s in enumerate(self._lb2id))

    def build_combination_search_space(self):
        """
            Build a search space for each possible combination of labels using bitmask
        """
        # Because we will build 2^n search spaces, n cannot be too large.
        assert len(self._labels) < 10, \
            f"N={len(self._labels)}, too large to build combination search spaces!"

        N = (1 << len(self._labels))
        comb_space = [[]] * N
        # Enumerate all the combinations of labels in a bitmask approach
        for i in range(N):
            for inst in self._insts:
                lbs_appeared = set()

    def build_single_label_search_space(self):
        pass

    def single_label_search_space(self, label:str) -> List[Instance]:
        pass

    def super_set_label_search_space() -> List[Instance]:
        pass