"""Tier-0 PoC: oversegment + average-linkage agglomeration decoder on V302 ABBC probs.
ABBC softmax channels: 0=bg, 1=border(outer surface), 2=boundary(inter-fragment fracture
surface), 3=core(eroded interior). Idea (GASP average-linkage / connectomics consensus):
 - oversegment the fragment foreground at EVERY boundary ridge (watershed on boundary-prob),
 - then conservatively agglomerate adjacent supervoxels whose shared interface boundary-prob
   is WEAK (avg-linkage), keeping only interfaces that look like true fracture surfaces (>= T).
This avoids the watershed MERGE (we split everywhere first) and the mutex OVER-split
(avg-linkage merges weak ridges back). Inference-only on the DEPLOYED V302 probs.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
try:
    from skimage.graph import rag_boundary, merge_hierarchical
except ImportError:
    from skimage.future.graph import rag_boundary, merge_hierarchical

# canonical skimage average-linkage merge functions for rag_boundary (weighted-mean boundary)
def _weight_boundary(graph, src, dst, n):
    d = {'weight': 0.0, 'count': 0}
    cs = graph[src].get(n, d)['count']; cd = graph[dst].get(n, d)['count']
    ws = graph[src].get(n, d)['weight']; wd = graph[dst].get(n, d)['weight']
    c = cs + cd
    return {'count': c, 'weight': (cs * ws + cd * wd) / c if c else 0.0}

def _merge_boundary(graph, src, dst):
    pass

def decode_agglo(probs, T=0.45, min_vox=250, seed_core=0.5, seed_bnd=0.20):
    """probs: float array [4,Z,Y,X]. Returns int instance map (0=bg)."""
    bg, border, boundary, core = probs[0], probs[1], probs[2], probs[3]
    fg = bg < 0.5
    if fg.sum() == 0:
        return np.zeros(fg.shape, np.int32)
    elev = boundary.astype(np.float32)  # high = fracture-surface ridge
    # oversegment: seed at strong-core / low-boundary interiors, watershed the boundary ridges
    seeds = (core > seed_core) & (boundary < seed_bnd) & fg
    markers, nm = ndi.label(seeds)
    if nm <= 1:  # fallback: seed at boundary-distance maxima
        d = ndi.distance_transform_edt(boundary < seed_bnd) * fg
        from skimage.feature import peak_local_max
        pk = peak_local_max(d, min_distance=4, labels=fg)
        m = np.zeros(fg.shape, bool); m[tuple(pk.T)] = True
        markers, nm = ndi.label(m)
        if nm == 0:
            markers, nm = ndi.label(fg)
    sv = watershed(elev, markers, mask=fg)
    if sv.max() <= 1:
        out = (sv > 0).astype(np.int32)
    else:
        rag = rag_boundary(sv, elev)
        if 0 in rag.nodes:          # drop the BACKGROUND node so fg regions can't agglomerate into bg
            rag.remove_node(0)
        merged = merge_hierarchical(sv, rag, thresh=T, rag_copy=False, in_place_merge=True,
                                    merge_func=_merge_boundary, weight_func=_weight_boundary)
        out = merged.astype(np.int32) + 1   # shift: merge_hierarchical is 0-based, so no fg region collides with bg=0
    out[~fg] = 0
    out, _ = _relabel(out)
    out = _drop_small(out, min_vox)
    return out

def _relabel(lab):
    ids = [i for i in np.unique(lab) if i != 0]
    remap = np.zeros(int(lab.max()) + 1, np.int32)
    for k, i in enumerate(ids, 1):
        remap[i] = k
    return remap[lab], len(ids)

def _drop_small(lab, min_vox):
    ids, cnts = np.unique(lab[lab > 0], return_counts=True)
    small = set(ids[cnts < min_vox].tolist())
    if not small:
        return lab
    keep = lab.copy()
    for s in small:
        keep[keep == s] = 0
    if (keep > 0).any() and (lab > 0).any():
        # reassign small voxels to nearest kept region
        idx = ndi.distance_transform_edt(keep == 0, return_distances=False, return_indices=True)
        filled = keep[tuple(idx)]
        m = (lab > 0) & (keep == 0)
        keep[m] = filled[m]
    lab2, _ = _relabel(keep)
    return lab2

def decode_affinity_agglo(abbc_probs, affinities, T=0.45, min_vox=250, short_idx=(0, 1, 2)):
    """[TIER-1] Decode V307's 13ch output into instances by average-linkage agglomeration on the
    LEARNED affinities (vs the noisy ABBC boundary channel that Tier-0 used).
      abbc_probs : [4,Z,Y,X] softmax (bg,border,boundary,core)
      affinities : [K,Z,Y,X] sigmoid, same-instance prob per loss.AFFINITY_HEAD_OFFSETS
    Separation ridge = 1 - min(short-range affinity): high where any short affinity is LOW = a true
    fracture surface. We splice this affinity-separation into the boundary channel and reuse the
    validated avg-linkage decoder (oversegment at ridges -> conservatively merge weak ridges)."""
    short = np.asarray(affinities)[list(short_idx)]
    sep = 1.0 - short.min(axis=0)                 # [Z,Y,X], high = fracture surface
    probs_aff = np.asarray(abbc_probs, dtype=np.float32).copy()
    probs_aff[2] = sep.astype(np.float32)         # replace ABBC boundary with the affinity separation
    return decode_agglo(probs_aff, T=T, min_vox=min_vox)


if __name__ == "__main__":
    import sys, glob, json, SimpleITK as sitk
    RANGES = {'Sacrum': (1, 50), 'LeftHip': (51, 100), 'RightHip': (101, 150), 'Femur': (151, 200)}
    cases = sys.argv[1:] or ["294", "256", "271", "286", "290", "021"]
    T = float(__import__('os').environ.get("AGGLO_T", "0.45"))
    def gtcount(c):
        g = glob.glob(f'/home/guest/Project/PENGWIN2026/data/task1_2/extracted/*/{c}/label.mha')
        a = sitk.GetArrayFromImage(sitk.ReadImage(g[0]))
        return {n: len([i for i in np.unique(a) if lo <= i <= hi]) for n, (lo, hi) in RANGES.items()
                if any(lo <= i <= hi for i in np.unique(a))}
    print(f"=== Tier-0 agglo decode (T={T}) vs GT fragment count (per anatomy) ===")
    for c in cases:
        gc = gtcount(c)
        probs_files = sorted(glob.glob(f'/tmp/probs/{c}/probs/probs_*.npy'))
        row = []
        for pf in probs_files:
            anat = pf.split('probs_')[-1].replace('.npy', '')
            p = np.load(pf).astype(np.float32)
            inst = decode_agglo(p, T=T)
            n = len([i for i in np.unique(inst) if i != 0])
            row.append(f"{anat}:agglo={n}/GT={gc.get(anat,'?')}")
        print(f"  case {c}: " + "  ".join(row))
