"""Unit-legality layer: make a semantic unit LEGAL FOR REMOVAL before editing.

Defenses (applied in order):
  D-floater  : after deleting the unit, any remainder component disconnected
               from the main body is pulled INTO the deletion set
               (feet-only-attached-to-legs, stray accessory cubes) —
               geometrically subsumes downward tree closure.
  D-symmetry : if the unit name is a symmetric/plural concept (wings, ears...),
               mirror the unit across the object's best symmetry plane and pull
               in clusters that match the mirror image (left wing -> right wing).
  D-gates    : final deletion share <= MAX_SHARE and remainder share >= MIN_REST;
               otherwise the unit is rejected (bad removal target).

All geometry, no VLM.  VLM verification (defense 2's Q&A) runs downstream on
the highlighted renders this module emits data for.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree

SYMMETRIC_VOCAB = {
    "wings", "wing", "ears", "ear", "horns", "horn", "arms", "arm", "legs",
    "leg", "eyes", "eye", "fins", "fin", "hands", "hand", "feet", "foot",
    "antennae", "antenna", "pipes", "side pipes", "wheels", "wheel",
    "shoulders", "boots", "gloves", "claws", "tusks", "exhausts",
}
MAX_SHARE = 0.55
MIN_REST = 0.30
ADJ_EPS_FRAC = 0.006      # adjacency: min point distance < 0.6% of bbox diag
SYM_MATCH_FRAC = 0.015    # mirror match: point within 1.5% diag
SYM_PULL_RATIO = 0.55     # pull pid if >55% of its points match the mirror


def _per_pid_points(meshes, fids, n=120000):
    concat = trimesh.util.concatenate(meshes)
    pts, fidx = trimesh.sample.sample_surface(concat, n)
    return np.asarray(pts), fids[np.asarray(fidx)], concat


def _areas_by_pid(meshes, fids):
    concat = trimesh.util.concatenate(meshes)
    fa = concat.area_faces
    out = {}
    for p in np.unique(fids):
        out[int(p)] = float(fa[fids == p].sum())
    return out, float(fa.sum())


def pid_adjacency(pts, ppid, diag):
    """proximity graph over pids (KDTree per pid, min-dist < eps)."""
    eps = diag * ADJ_EPS_FRAC
    pids = np.unique(ppid)
    cloud = {p: pts[ppid == p] for p in pids}
    trees = {p: cKDTree(cloud[p][::4] if len(cloud[p]) > 400 else cloud[p])
             for p in pids}
    adj = {int(p): set() for p in pids}
    plist = list(pids)
    for i, a in enumerate(plist):
        qa = cloud[a][::8] if len(cloud[a]) > 800 else cloud[a]
        for b in plist[i + 1:]:
            d, _ = trees[b].query(qa, k=1, distance_upper_bound=eps)
            if np.isfinite(d).any():
                adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    return adj


def _components(adj, keep):
    seen, comps = set(), []
    for s in keep:
        if s in seen:
            continue
        stack, comp = [s], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x); seen.add(x)
            stack.extend(adj[x] & keep)
        comps.append(comp)
    return comps


def floater_closure(adj, areas, unit):
    keep = set(adj) - set(unit)
    comps = _components(adj, keep)
    if not comps:
        return set(unit), []
    main = max(comps, key=lambda c: sum(areas[p] for p in c))
    pulled = [p for c in comps if c is not main for p in c]
    return set(unit) | set(pulled), pulled


def best_mirror_plane(pts, diag):
    """axis-aligned plane through bbox center with best self-symmetry."""
    c = (pts.min(0) + pts.max(0)) / 2
    best = (1e9, 0)
    tree = cKDTree(pts[::5])
    for ax in range(3):
        m = pts[::7].copy()
        m[:, ax] = 2 * c[ax] - m[:, ax]
        d, _ = tree.query(m, k=1)
        score = np.median(d) / diag
        if score < best[0]:
            best = (score, ax)
    return best[1], c, best[0]


def symmetry_closure(pts, ppid, unit, name, diag):
    if name.strip().lower() not in SYMMETRIC_VOCAB:
        return set(unit), [], None
    ax, c, sym_q = best_mirror_plane(pts, diag)
    if sym_q > 0.01:                      # object itself not symmetric
        return set(unit), [], f"asym(q={sym_q:.3f})"
    upts = pts[np.isin(ppid, list(unit))]
    m = upts.copy(); m[:, ax] = 2 * c[ax] - m[:, ax]
    mtree = cKDTree(m)
    pulled = []
    for p in np.unique(ppid):
        p = int(p)
        if p in unit:
            continue
        q = pts[ppid == p]
        d, _ = mtree.query(q[::3] if len(q) > 300 else q, k=1)
        if (d < diag * SYM_MATCH_FRAC).mean() > SYM_PULL_RATIO:
            pulled.append(p)
    return set(unit) | set(pulled), pulled, f"axis={'xyz'[ax]}"


def legalize(meshes, fids, unit_pids, name):
    """Returns dict(final_pids, pulled_floaters, pulled_symmetry, share,
    rest_share, verdict, notes)."""
    pts, ppid, concat = _per_pid_points(meshes, fids)
    diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
    areas, total = _areas_by_pid(meshes, fids)
    unit = set(int(p) for p in unit_pids)

    unit, pulled_sym, sym_note = symmetry_closure(pts, ppid, unit, name, diag)
    adj = pid_adjacency(pts, ppid, diag)
    unit, pulled_flt = floater_closure(adj, areas, unit)
    # symmetric partners may create new floaters -> one more pass
    unit, pulled_flt2 = floater_closure(adj, areas, unit)
    pulled_flt += pulled_flt2

    share = sum(areas.get(p, 0) for p in unit) / total
    rest = 1 - share
    verdict = "ok"
    if share > MAX_SHARE: verdict = f"reject:share>{MAX_SHARE}"
    elif rest < MIN_REST: verdict = f"reject:rest<{MIN_REST}"
    return {
        "final_pids": sorted(unit), "pulled_symmetry": sorted(pulled_sym),
        "pulled_floaters": sorted(set(pulled_flt)), "share": round(share, 3),
        "rest_share": round(rest, 3), "verdict": verdict,
        "sym_note": sym_note,
    }
