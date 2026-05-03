"""Utility helpers (I/O, seeds)."""
import pickle
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
