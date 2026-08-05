"""Run nnUNet's resampling in float32 instead of float64.

Why. A 7-case peak-RSS ladder (11.9M .. 131.9M voxels) put the shipped container at 8.03 GB on the
largest volume, and an object-level probe named where it sits. Two of the largest resident arrays
are float64 copies of full-resolution data created by third-party resampling code:

  nnunetv2/preprocessing/resampling/default_resampling.py:144
      data = data.astype(float, copy=False)          # `float` is float64
      -> np(1, 592, 435, 512) float64 = 1.05 GB on case 285

  batchgenerators/augmentations/utils.py:604  (inside resize_segmentation)
      resize(mask.astype(float), ...)                # another full-size float64, per label
      -> a second 1.05 GB transient, allocated once per unique label

The inputs are a segmentation (int8 here) and a boolean one-hot mask. float32 represents every
integer up to 2^24 exactly, so for label data this is not an approximation at all; the only place
precision can matter is the spline interpolation feeding the `>= 0.5` one-hot threshold, where the
float32/float64 difference is ~1e-7 against a threshold of 0.5. A voxel would have to land within
1e-7 of exactly 0.5 to flip. For scale, GPU inference on this pipeline is not bit-reproducible and
already moves ~59 voxels of 132M between identical runs, which is why sha256 is not a valid QA
instrument here and why result-identity is checked on labels, fragment counts and volumes instead.

Installed by inference.py and gated on PENGWIN_LOWMEM_RESAMPLE (default on). Set it to 0 to fall
back to the stock float64 path, which is the control arm for any QA comparison.

Both functions are byte-for-byte the upstream implementations with `float` -> `np.float32` and the
untyped `np.zeros(tmp)` given an explicit float32 dtype. They are pinned to nnunetv2 2.5.1 /
batchgenerators as vendored in this image; install() verifies the upstream source still matches what
was copied and REFUSES to patch if it has changed, rather than silently running a stale copy.
"""
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from skimage.transform import resize


def resize_segmentation_f32(segmentation, new_shape, order=3):
    """batchgenerators.augmentations.utils.resize_segmentation with float32 one-hot masks."""
    tpe = segmentation.dtype
    assert len(segmentation.shape) == len(new_shape), "new shape must have same dimensionality as segmentation"
    if order == 0:
        return resize(segmentation.astype(np.float32), new_shape, order, mode="edge",
                      clip=True, anti_aliasing=False).astype(tpe)
    reshaped = np.zeros(new_shape, dtype=segmentation.dtype)
    unique_labels = np.sort(pd.unique(segmentation.ravel()))
    for c in unique_labels:
        mask = segmentation == c
        reshaped_multihot = resize(mask.astype(np.float32), new_shape, order, mode="edge",
                                   clip=True, anti_aliasing=False)
        reshaped[reshaped_multihot >= 0.5] = c
    return reshaped


def resample_data_or_seg_f32(data, new_shape, is_seg=False, axis=None, order=3,
                             do_separate_z=False, order_z=0, dtype_out=None):
    """nnunetv2 default_resampling.resample_data_or_seg with the float64 working copy in float32."""
    assert data.ndim == 4, "data must be (c, x, y, z)"
    assert len(new_shape) == data.ndim - 1

    if is_seg:
        resize_fn = resize_segmentation_f32
        kwargs = OrderedDict()
    else:
        resize_fn = resize
        kwargs = {'mode': 'edge', 'anti_aliasing': False}
    shape = np.array(data[0].shape)
    new_shape = np.array(new_shape)
    if dtype_out is None:
        dtype_out = data.dtype
    reshaped_final = np.zeros((data.shape[0], *new_shape), dtype=dtype_out)
    if np.any(shape != new_shape):
        data = data.astype(np.float32, copy=False)
        if do_separate_z:
            assert axis is not None, 'If do_separate_z, we need to know what axis is anisotropic'
            if axis == 0:
                new_shape_2d = new_shape[1:]
            elif axis == 1:
                new_shape_2d = new_shape[[0, 2]]
            else:
                new_shape_2d = new_shape[:-1]

            for c in range(data.shape[0]):
                tmp = deepcopy(new_shape)
                tmp[axis] = shape[axis]
                reshaped_here = np.zeros(tmp, dtype=np.float32)   # upstream leaves this float64
                for slice_id in range(shape[axis]):
                    if axis == 0:
                        reshaped_here[slice_id] = resize_fn(data[c, slice_id], new_shape_2d, order, **kwargs)
                    elif axis == 1:
                        reshaped_here[:, slice_id] = resize_fn(data[c, :, slice_id], new_shape_2d, order, **kwargs)
                    else:
                        reshaped_here[:, :, slice_id] = resize_fn(data[c, :, :, slice_id], new_shape_2d, order, **kwargs)
                if shape[axis] != new_shape[axis]:
                    rows, cols, dim = new_shape[0], new_shape[1], new_shape[2]
                    orig_rows, orig_cols, orig_dim = reshaped_here.shape

                    row_scale = float(orig_rows) / rows
                    col_scale = float(orig_cols) / cols
                    dim_scale = float(orig_dim) / dim

                    map_rows, map_cols, map_dims = np.mgrid[:rows, :cols, :dim]
                    map_rows = row_scale * (map_rows + 0.5) - 0.5
                    map_cols = col_scale * (map_cols + 0.5) - 0.5
                    map_dims = dim_scale * (map_dims + 0.5) - 0.5

                    coord_map = np.array([map_rows, map_cols, map_dims])
                    if not is_seg or order_z == 0:
                        reshaped_final[c] = map_coordinates(reshaped_here, coord_map, order=order_z, mode='nearest')[None]
                    else:
                        unique_labels = np.sort(pd.unique(reshaped_here.ravel()))
                        for cl in unique_labels:
                            reshaped_final[c][np.round(
                                map_coordinates((reshaped_here == cl).astype(np.float32), coord_map,
                                                order=order_z, mode='nearest')) > 0.5] = cl
                else:
                    reshaped_final[c] = reshaped_here
        else:
            for c in range(data.shape[0]):
                reshaped_final[c] = resize_fn(data[c], new_shape, order, **kwargs)
        return reshaped_final
    return data


# The exact upstream lines this file replaces. If a dependency bump changes them, the copy above is
# stale and could silently diverge from what the rest of nnUNet expects, so refuse to install.
_EXPECTED = (
    ("nnunetv2.preprocessing.resampling.default_resampling", "resample_data_or_seg",
     "data = data.astype(float, copy=False)"),
    ("batchgenerators.augmentations.utils", "resize_segmentation",
     "reshaped_multihot = resize(mask.astype(float), new_shape, order, mode=\"edge\", clip=True, anti_aliasing=False)"),
)


def install(log=print):
    """Patch the float64 resamplers. Returns True if patched, False if left alone."""
    import importlib
    import inspect

    for mod_name, fn_name, needle in _EXPECTED:
        try:
            src = inspect.getsource(getattr(importlib.import_module(mod_name), fn_name))
        except Exception as exc:  # noqa: BLE001
            log(f"lowmem-resample: cannot read {mod_name}.{fn_name} ({exc}); NOT patching")
            return False
        if needle not in src:
            log(f"lowmem-resample: {mod_name}.{fn_name} no longer matches the vendored copy; NOT patching")
            return False

    dr = importlib.import_module("nnunetv2.preprocessing.resampling.default_resampling")
    # resample_data_or_seg_to_shape / _to_spacing call it as a module global, so patching the
    # attribute is enough for every path our inference uses.
    dr.resample_data_or_seg = resample_data_or_seg_f32
    dr.resize_segmentation = resize_segmentation_f32
    log("lowmem-resample: nnUNet resampling now runs in float32")
    return True
