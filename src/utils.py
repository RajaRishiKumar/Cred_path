from pathlib import Path
import joblib


# Project root

PROJECT_ROOT = Path(__file__).resolve().parents[1]



# Generic helpers


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path



# Artifacts (GLOBAL)

ARTIFACTS_DIR = ensure_dir(PROJECT_ROOT / "artifacts")
MODELS_DIR = ensure_dir(ARTIFACTS_DIR / "models")



# Save / Load


def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.pkl")


def save_artifacts(obj, name: str):
    path = ARTIFACTS_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    return path


def load_artifacts(name: str):
    return joblib.load(ARTIFACTS_DIR / f"{name}.joblib")


def print_tree(path, depth=3, prefix=""):
    if depth < 0:
        return
    print(prefix + path.name + "/")
    for p in path.iterdir():
        if p.is_dir():
            print_tree(p, depth - 1, prefix + "    ")
        else:
            print(prefix + "    " + p.name)
