from dataclasses import dataclass
from typing import List

@dataclass
class Instance:
	words: List[str]
	ori_words: List[str]
	labels: List[str] = None
	final_labels: List[str] = None
	prediction: List[str]  = None
	trigger_positions: List[int] = None
	entities: List[str] = None
	triggers: List[str] = None

