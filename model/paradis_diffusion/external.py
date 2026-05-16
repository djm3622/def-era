"""Load PARADIS modules from the git submodule without renaming this package.

The upstream PARADIS repository imports its internals as ``model.*``. This repo
also has a top-level ``model`` package, so importing PARADIS directly would make
the two packages collide. The loader below temporarily exposes the submodule's
``model`` package just long enough to load the PARADIS class, then restores this
repo's package namespace.
"""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Union


def load_paradis_class(paradis_root: Union[str, Path] = "paradis"):
    """Return the upstream ``Paradis`` class from the checked-out submodule."""

    root = Path(paradis_root).resolve()
    model_dir = root / "model"
    paradis_file = model_dir / "paradis.py"

    if not paradis_file.exists():
        raise FileNotFoundError(
            f"Could not find {paradis_file}. Initialize the submodule with "
            "`git submodule update --init paradis`."
        )

    saved_model = sys.modules.get("model")
    saved_submodules = {
        name: sys.modules.get(name)
        for name in (
            "model.padding",
            "model.blocks",
            "model.advection",
            "model.paradis",
        )
    }

    try:
        shim = types.ModuleType("model")
        shim.__path__ = [str(model_dir)]
        sys.modules["model"] = shim
        for name in saved_submodules:
            sys.modules.pop(name, None)

        spec = importlib.util.spec_from_file_location("model.paradis", paradis_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for {paradis_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules["model.paradis"] = module
        spec.loader.exec_module(module)
        return module.Paradis
    finally:
        for name, module in saved_submodules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

        if saved_model is None:
            sys.modules.pop("model", None)
        else:
            sys.modules["model"] = saved_model
