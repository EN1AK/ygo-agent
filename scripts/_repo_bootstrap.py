from pathlib import Path
import sys


def add_repo_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    scripts_dir = Path(__file__).resolve().parent
    paths = [
        root / "ygoenv",
        root / "ygoinf",
        root / "mcts",
        root,
    ]
    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    scripts_dir_str = str(scripts_dir)
    while scripts_dir_str in sys.path:
        sys.path.remove(scripts_dir_str)


add_repo_paths()
