from importlib import import_module
from inspect import getmembers, isclass, isfunction, ismodule
from pathlib import Path

recipes = {}

recipes_dir = Path(__file__).parent / "recipes"
if recipes_dir.is_dir():
    for f in recipes_dir.glob("*.py"):
        if f.name == "__init__.py":
            continue
        try:
            m = import_module(f"{__name__}.recipes.{f.stem}")
        except Exception as e:
            print(f"Warning: Could not import {f.stem}: {e}")
            continue
        for name, obj in getmembers(m):
            if name.startswith("_") or ismodule(obj):
                continue
            if (isclass(obj) or isfunction(obj)) and getattr(obj, "__module__", None) == m.__name__:
                recipes[name] = obj

__all__ = ["recipes"]
