"""[v3.9 mem] Keep TRAINING-only third-party libraries out of the inference process.

Why this exists
---------------
`import nnunetv2.inference.predict_from_raw_data` costs ~0.23 GiB of RESIDENT RSS, and a
measurable part of that is libraries a Grand-Challenge inference run never touches.  They are
pulled in at *module import time*, deep inside nnU-Net 2.5.1, so there is no nnU-Net-level flag
to turn them off:

    matplotlib + seaborn   <- nnunetv2/training/logging/nnunet_logger.py:1-6
                              (imported by nnUNetTrainer.py, which our trainer subclasses; it
                              exists only for plot_progress_png -> training only)
    nibabel                <- nnunetv2/imageio/nibabel_reader_writer.py:18,21, imported
                              unconditionally by nnunetv2/imageio/reader_writer_registry.py.
                              nibabel in turn drags in pydicom -> requests -> urllib3.
                              Both our datasets declare `"file_ending": ".mha"`, so nnU-Net
                              selects SimpleITKIO and NibabelIO is never instantiated.

Resident memory is the only kind that actually lowers the end-to-end peak: the peak lands late
(Stage-B build_predictor) and glibc never returns freed blocks, so a *transient* saving early is
worth nothing while a resident saving subtracts 1:1.  These imports are resident for the whole
run.

How it works
------------
We pre-register LAZY STUB modules in `sys.modules` before nnU-Net is imported.  The stubs are not
inert: the first time anything reads an attribute the real package is imported for real and the
stub starts proxying it.  So the worst case of a wrong assumption is that we pay the memory we
were trying to save -- never a wrong answer and never an ImportError.  `PENGWIN_LEAN_IMPORTS=0`
disables the whole thing; `PENGWIN_LEAN_IMPORTS_SKIP=mpl,nib` disables individual groups.

Deliberately NOT stubbed
------------------------
* torch / nnunetv2 / SimpleITK / scipy / numpy  -- the hot path.
* sklearn + joblib -- the Stage-1 target-family RandomForest router unpickles on EVERY case, so
  they are hot-path too and a stub would only move the cost, not remove it.
* pandas -- nnU-Net's own `default_resampling.resample_data_or_seg_to_shape` calls `pd.unique`
  on the is_seg=True path, which inference.py DOES use; a stub materialises immediately (measured
  saving 0.000 GiB) and replacing pd.unique with np.unique would be a hot-path behaviour change.
"""
from __future__ import annotations

import os
import sys
import types

# Attributes that must never trigger the real import: the import machinery, pickle, copy and
# inspect probe these on arbitrary objects, and materialising there would defeat the point.
_NEVER = frozenset({
    "__all__", "__file__", "__spec__", "__loader__", "__package__", "__path__",
    "__warningregistry__", "__getstate__", "__setstate__", "__reduce__",
    "__reduce_ex__", "__copy__", "__deepcopy__", "__wrapped__", "__bases__",
    "__class_getitem__", "_ipython_canary_method_should_not_exist_",
})

# name -> reason, filled in when a stub is materialised (so a run can report what it got wrong)
MATERIALISED: dict[str, str] = {}
INSTALLED: list[str] = []


class _LazyStub(types.ModuleType):
    """A placeholder module that imports the real package on first attribute access."""

    def __init__(self, fullname: str, group: tuple[str, ...]):
        super().__init__(fullname)
        d = self.__dict__
        d["_pengwin_fullname"] = fullname
        d["_pengwin_group"] = group
        d["_pengwin_real"] = None
        # __path__ makes `import pkg.sub` accept us as a package; the submodules we care about
        # are pre-registered in sys.modules so the finder never walks this (empty) path.
        d["__path__"] = []

    def __getattr__(self, item):
        if item in _NEVER:
            raise AttributeError(item)
        real = self.__dict__.get("_pengwin_real")
        if real is None:
            real = _materialise(self.__dict__["_pengwin_fullname"],
                                self.__dict__["_pengwin_group"], item)
        return getattr(real, item)


def _materialise(fullname: str, group: tuple[str, ...], trigger: str):
    """Drop every stub of this package from sys.modules and import it for real."""
    import importlib

    for name in group:
        mod = sys.modules.get(name)
        if isinstance(mod, _LazyStub):
            del sys.modules[name]
    real_top = importlib.import_module(group[0])
    # Re-point every stub object at its real counterpart so already-bound references keep working.
    for name in group:
        try:
            real = importlib.import_module(name)
        except Exception:  # noqa: BLE001  - an optional submodule that does not exist
            continue
        stub = _STUBS.get(name)
        if stub is not None:
            stub.__dict__["_pengwin_real"] = real
    MATERIALISED[fullname] = trigger
    real = sys.modules.get(fullname) or real_top
    _log(f"lean-imports: MATERIALISED {fullname} (triggered by .{trigger}) - no saving from it")
    return real


class _LazyFunc:
    """A callable placeholder for `from pkg import name` at *module* level.

    `from nibabel import io_orientation` reads the attribute during import, which would
    materialise the package immediately.  Handing back one of these defers the real import to the
    first CALL -- and nnU-Net only ever calls io_orientation inside NibabelIOWithReorient.write_seg,
    which a .mha dataset never reaches.
    """

    __slots__ = ("_fullname", "_group", "_attr")

    def __init__(self, fullname: str, group: tuple[str, ...], attr: str):
        self._fullname = fullname
        self._group = group
        self._attr = attr

    def __call__(self, *args, **kwargs):
        real = _materialise(self._fullname, self._group, self._attr + "()")
        return getattr(real, self._attr)(*args, **kwargs)

    def __repr__(self):
        return f"<pengwin lazy {self._fullname}.{self._attr}>"


_STUBS: dict[str, _LazyStub] = {}


def _log(msg: str) -> None:
    print(f"[pengwin] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Groups.  Each entry: key -> (tuple of module names to stub, post-install tweak)
# The FIRST name in the tuple is the top-level package.
# ---------------------------------------------------------------------------
_GROUPS: dict[str, tuple[str, ...]] = {
    # matplotlib.use('agg') is called at import time by nnunet_logger; see _tweak_mpl.
    "mpl": ("matplotlib", "matplotlib.pyplot", "matplotlib.cm", "matplotlib.colors"),
    "sns": ("seaborn",),
    "nib": ("nibabel",),
}


def _tweak_mpl(stub: _LazyStub) -> None:
    """`nnunet_logger` calls matplotlib.use('agg') at import time.

    Answering that from the stub is what keeps matplotlib unimported.  Selecting a backend has no
    observable effect unless something actually plots, and if something ever does, the recorded
    backend is replayed onto the real matplotlib at materialisation time.
    """
    def use(backend, *a, **k):
        stub.__dict__["_pengwin_backend"] = backend
        real = stub.__dict__.get("_pengwin_real")
        if real is not None:
            real.use(backend, *a, **k)

    stub.__dict__["use"] = use
    stub.__dict__["get_backend"] = lambda: stub.__dict__.get("_pengwin_backend", "agg")


def _tweak_nib(stub: _LazyStub) -> None:
    """`nibabel_reader_writer` does `from nibabel import io_orientation` at module level."""
    group = stub.__dict__["_pengwin_group"]
    for attr in ("io_orientation",):
        stub.__dict__[attr] = _LazyFunc("nibabel", group, attr)


_TWEAKS = {"matplotlib": _tweak_mpl, "nibabel": _tweak_nib}


def install(verbose: bool = True) -> list[str]:
    """Register the stubs.  Safe to call more than once; a no-op for anything already imported."""
    if _STUBS:  # already installed (inference.py may be imported twice by the dev harness)
        return list(INSTALLED)
    if os.environ.get("PENGWIN_LEAN_IMPORTS", "1") != "1":
        if verbose:
            _log("lean-imports: disabled by PENGWIN_LEAN_IMPORTS=0")
        return []
    skip = {s.strip() for s in os.environ.get("PENGWIN_LEAN_IMPORTS_SKIP", "").split(",") if s.strip()}
    done = []
    for key, names in _GROUPS.items():
        if key in skip:
            continue
        top = names[0]
        if top in sys.modules:  # already imported for real -> too late, leave it alone
            continue
        parent = None
        for name in names:
            stub = _LazyStub(name, names)
            _STUBS[name] = stub
            sys.modules[name] = stub
            tweak = _TWEAKS.get(name)
            if tweak is not None:
                tweak(stub)
            if parent is None:
                parent = stub
            else:
                # Pre-wire `pkg.sub` as an ATTRIBUTE of the parent stub, otherwise
                # `import pkg.sub as x` reads it through __getattr__ and materialises the package.
                parent.__dict__[name.split(".", 1)[1]] = stub
        done.append(key)
    INSTALLED[:] = done
    if verbose and done:
        _log(f"lean-imports: deferred {', '.join(done)} "
             f"({sum(len(_GROUPS[k]) for k in done)} module slots)")
    return done


def report() -> str:
    if not INSTALLED:
        return "lean-imports: nothing deferred"
    bad = ", ".join(sorted(MATERIALISED)) or "none"
    return f"lean-imports: deferred={','.join(INSTALLED)} materialised={bad}"
