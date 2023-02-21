import pathlib


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent
