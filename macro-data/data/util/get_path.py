from pathlib import Path


def get_main_path() -> Path:
    return Path(__file__).parent.parent.parent
