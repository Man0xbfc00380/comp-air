"""Prep sys.path for the `booksim2` pybind module (`booksim2/api/build`) and `booksim_sync` (`booksim2/api`)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", ".."))
_api_dir = os.path.join(_repo_root, "booksim2", "api")
_api_build = os.path.join(_api_dir, "build")
if _api_build not in sys.path:
    sys.path.insert(0, _api_build)
if _api_dir not in sys.path:
    sys.path.insert(0, _api_dir)
