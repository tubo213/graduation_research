import pickle
from pathlib import Path


def save_as_pickle(obj, path: Path):
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)
