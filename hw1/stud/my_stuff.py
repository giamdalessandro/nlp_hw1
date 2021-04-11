import os
import time
import jsonlines
import numpy as np

from tqdm import tqdm
from typing import List, Tuple, Any, Dict

"""
    Using the provided functions to load the data, as in evaluate.py:
        - count(l: List[Any]) -> Dict[Any, int];
        - read_dataset(path: str) -> Tuple[List[Dict], List[str]]; 
"""
def count(l: List[Any]) -> Dict[Any, int]:
    d = {}
    for e in l:
        d[e] = 1 + d.get(e, 0)
    return d

def read_dataset(path: str) -> Tuple[List[Dict], List[str]]:

    sentence_pairs = []
    labels = []

    with jsonlines.open(path) as f:
        for obj in f:
            labels.append(obj.pop('label'))
            sentence_pairs.append(obj)

    assert len(sentence_pairs) == len(labels)

    return sentence_pairs, labels



if __name__ == '__main__':
    #os.chdir("../../")
    data_path = "data/dev.jsonl"

    try:
        sentence_pairs, labels = read_dataset(data_path)

    except FileNotFoundError as e:
        print(f'Evaluation crashed because {data_path} does not exist')
        exit(1)

    except Exception as e:
        print(f'Evaluation crashed. Most likely, the file you gave is not in the correct format')
        print(f'Printing error found')
        print(e, exc_info=True)
        exit(1)
    
    print("labels:",len(labels))
    print("sentence pairse:",len(sentence_pairs))
    print("[INFO]: data loaded successfully.")