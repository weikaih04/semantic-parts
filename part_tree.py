"""Semantic part grouping on top of P3-SAM (or any flat part segmentation).

P3-SAM gives you a FLAT set of parts. That is not what an edit / articulation
unit is: an eye lives *inside* a head, a head is often several nested shells,
and one smooth surface frequently gets cut into two ids. Move the "head" and
the eyes stay behind.

This module adds the geometric layer that recovers those relations. It is
deliberately VLM-free — it handles exactly what a VLM cannot SEE (hidden
interiors, double walls, seams), and leaves naming/semantics to whatever you
put on top.

Three mechanisms
----------------
1. containment (generalized winding number)
     part Q becomes a CHILD of bigger part P when >= INSIDE_FRAC of Q's surface
     samples have |w(P, q)| > 0.5.  The GENERALIZED winding number is the point:
     it stays meaningful for OPEN / non-watertight shells, which is the normal
     case for scanned + game assets (a head that is an open shell still
     "contains" the eyeballs).
2. nested-shell clustering
     semantic solids are often several hugging shells of comparable size. Merge
     P,Q when the smaller hugs the bigger (contact fraction) or their bboxes
     overlap strongly. A head touching a body only at a neck ring does NOT merge
     (low contact), so real joints survive.
3. tangent-continuity seam merge
     if two parts meet along a rim where normals continue smoothly (median angle
     < SEAM_DEG) they are two halves of ONE surface -> merge. A part sitting ON
     another meets at a concave crease (normals disagree) -> keep separate.

Output
------
build_part_tree(vertices, faces, face_ids) -> dict
    tree    {pid: {"parent": pid|-1, "rule": "inside"|"attached"|None, "score": f}}
    cluster {pid: cluster_root_pid}          # co-moving shells
    group   {pid: [pid, ...]}                # THE EDIT UNIT: part + descendants
                                             # + its cluster. Move/delete this set.

Deps: numpy, scipy, trimesh, libigl (`pip install libigl`)
Cost: ~1-3 s per asset at NSAMP=40k on CPU; winding number dominates.
"""
import numpy as np
import trimesh
import igl
from scipy.spatial import cKDTree

# --- tunables (values below are what we shipped; all are conservative) -------
INSIDE_FRAC = 0.55    # frac of Q's samples inside P  -> containment
CONTACT_FRAC = 0.45   # frac of Q's samples on P's surface -> attachment
SIZE_RATIO = 0.25     # attachment only if area(Q) <= SIZE_RATIO * area(P)
CLUSTER_CONTACT = 0.35
CLUSTER_BBOX_IOU = 0.50
SEAM_DEG = 22.0
EPS_FRAC = 0.02       # "on the surface" = within 2% of bbox diagonal
NSAMP = 40000


def _find(uf, u):
    while uf[u] != u:
        uf[u] = uf[uf[u]]
        u = uf[u]
    return u


def _bbox_iou_small(A, B):
    lo = np.maximum(A["lo"], B["lo"]); hi = np.minimum(A["hi"], B["hi"])
    inter = np.prod(np.maximum(hi - lo, 0))
    va = np.prod(A["hi"] - A["lo"]); vb = np.prod(B["hi"] - B["lo"])
    return inter / max(min(va, vb), 1e-12)      # relative to the SMALLER box


def build_part_tree(vertices, faces, face_ids, nsamp=NSAMP, seed=0):
    """vertices (V,3) float, faces (F,3) int, face_ids (F,) int  [P3-SAM output].
    Returns {"tree","cluster","group"} keyed by str(pid)."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int64)
    fids = np.asarray(face_ids)
    merged = trimesh.Trimesh(vertices=V, faces=F, process=False)
    areas = merged.area_faces
    diag = float(np.linalg.norm(np.ptp(V, axis=0)))
    eps = EPS_FRAC * diag

    np.random.seed(seed)
    pts, fi = trimesh.sample.sample_surface(merged, nsamp)
    ppid = fids[fi]
    fnrm = merged.face_normals[fi]

    info = {}
    for u in np.unique(fids):
        m = ppid == u
        if m.sum() < 30:                       # too few samples to judge
            continue
        p = pts[m]
        info[int(u)] = {"area": float(areas[fids == u].sum()), "pts": p,
                        "nrm": fnrm[m], "lo": p.min(0), "hi": p.max(0)}
    if not info:
        return {"tree": {}, "cluster": {}, "group": {}}
    ids = sorted(info, key=lambda u: -info[u]["area"])     # big -> small

    # ---- 2+3: cluster similar-sized shells (hug / nested / smooth seam) -----
    uf = {u: u for u in info}
    for i, p in enumerate(ids):
        for q in ids[i + 1:]:
            P, Q = info[p], info[q]
            if Q["area"] < SIZE_RATIO * P["area"]:
                continue                        # small ones go to the tree instead
            qs = Q["pts"][::max(1, len(Q["pts"]) // 600)]
            qn = Q["nrm"][::max(1, len(Q["nrm"]) // 600)]
            d, idx = cKDTree(P["pts"]).query(qs, k=1)
            near = d < eps
            if float(near.mean()) >= CLUSTER_CONTACT or _bbox_iou_small(P, Q) >= CLUSTER_BBOX_IOU:
                uf[_find(uf, p)] = _find(uf, q)
                continue
            if near.sum() >= 25:                # smooth seam => one surface
                ang = np.degrees(np.arccos(np.clip(
                    np.abs((qn[near] * P["nrm"][idx[near]]).sum(1)), 0, 1)))
                if np.median(ang) < SEAM_DEG:
                    uf[_find(uf, p)] = _find(uf, q)
    cluster_of = {u: _find(uf, u) for u in info}

    # ---- 1: containment / attachment tree ----------------------------------
    tree = {u: {"parent": -1, "rule": None, "score": 0.0} for u in info}
    for qi, q in enumerate(ids):
        Q = info[q]
        qs = Q["pts"][::max(1, len(Q["pts"]) // 800)]
        best = None
        for p in ids[:qi]:                      # only strictly bigger parts
            P = info[p]
            c = (Q["lo"] + Q["hi"]) / 2         # cheap bbox prefilter
            slack = 0.10 * (P["hi"] - P["lo"] + 1e-9)
            if not np.all((c > P["lo"] - slack) & (c < P["hi"] + slack)):
                continue
            Fp = F[fids == p]
            w = igl.winding_number(V, np.asarray(Fp, dtype=np.int64),
                                   np.asarray(qs, dtype=np.float64))
            in_frac = float((np.abs(w) > 0.5).mean())
            if in_frac >= INSIDE_FRAC:
                cand = ("inside", in_frac, p)
            elif Q["area"] <= SIZE_RATIO * P["area"]:
                d, _ = cKDTree(P["pts"]).query(qs, k=1)
                cf = float((d < eps).mean())
                cand = ("attached", cf, p) if cf >= CONTACT_FRAC else None
            else:
                cand = None
            if cand and (best is None or cand[1] > best[1]):
                best = cand
        if best:
            tree[q] = {"parent": int(best[2]), "rule": best[0], "score": round(best[1], 3)}

    # ---- edit groups: cluster members + everything hanging under them -------
    kids = {}
    for c, v in tree.items():
        if v["parent"] != -1:
            kids.setdefault(cluster_of[v["parent"]], []).append(c)
    groups = {}
    for u in info:
        root = cluster_of[u]
        members = [x for x in info if cluster_of[x] == root]
        stack, seen = list(members), set(members)
        while stack:
            n = stack.pop()
            for ch in kids.get(cluster_of[n], []):
                if ch not in seen:
                    seen.add(ch); stack.append(ch)
            for ch, v in tree.items():
                if v["parent"] == n and ch not in seen:
                    seen.add(ch); stack.append(ch)
        groups[u] = sorted(int(x) for x in seen)

    return {"tree": {str(k): v for k, v in tree.items()},
            "cluster": {str(k): int(v) for k, v in cluster_of.items()},
            "group": {str(k): v for k, v in groups.items()}}


def subtree(tree, root):
    """all descendants of `root` (inclusive), from a returned tree dict."""
    kids = {}
    for c, v in tree.items():
        kids.setdefault(int(v["parent"]), []).append(int(c))
    out, stack = [], [int(root)]
    while stack:
        n = stack.pop()
        out.append(n)
        stack += kids.get(n, [])
    return out
