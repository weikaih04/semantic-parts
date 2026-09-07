"""End-to-end: P3-SAM face_ids -> named, edit-legal semantic parts.

    # CPU
    python pipeline.py geom mesh.glb face_ids.npy outdir/ [--photo render.png]
    # GPU (vLLM)
    python pipeline.py name outdir/ [outdir2/ ...]
    # CPU, optional: make one named unit legal to DELETE
    python pipeline.py legalize outdir/ "left wing"

Stages
  geom     part_tree (containment / shells / seams) + colour panel
  name     one VLM call -> {object, groups[{name, colors}]} -> units.json
  legalize symmetry + floater closure + share gates on ONE named unit
"""
import os, sys, json
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from part_tree import build_part_tree
from color_panel import build_panel


def load_mesh(glb):
    sc = trimesh.load(glb)
    return (trimesh.util.concatenate(
                [g for g in sc.geometry.values() if isinstance(g, trimesh.Trimesh)])
            if isinstance(sc, trimesh.Scene) else sc)


def geom(glb, fid_path, out, photo=None):
    mesh = load_mesh(glb)
    fids = np.load(fid_path)
    assert len(fids) == len(mesh.faces), (
        f"face_ids ({len(fids)}) != faces ({len(mesh.faces)}). Load the mesh with the SAME "
        "loader that produced face_ids — see README 'provenance'.")
    os.makedirs(out, exist_ok=True)
    pt = build_part_tree(mesh.vertices, mesh.faces, fids)
    json.dump(pt, open(f"{out}/part_tree.json", "w"), indent=1)
    colors = build_panel(mesh, fids, pt, out, extra_image=photo)
    t = pt["tree"]
    nin = sum(1 for v in t.values() if v["rule"] == "inside")
    nat = sum(1 for v in t.values() if v["rule"] == "attached")
    print(f"{len(t)} parts -> {len(set(pt['cluster'].values()))} clusters "
          f"({nin} inside, {nat} attached) -> {len(colors)} colours\n  {out}/panel.jpg")


def name(dirs):
    from vlm_group import build_llm, ask, resolve_groups
    items, keep = [], []
    for d in dirs:
        p, c = f"{d}/panel.jpg", f"{d}/colors.json"
        if os.path.exists(p) and os.path.exists(c):
            items.append((p, json.load(open(c)))); keep.append(d)
    if not items:
        print("nothing to name"); return
    llm = build_llm()
    for d, colors, out in zip(keep, [i[1] for i in items], ask(llm, items)):
        units = resolve_groups(out, colors)
        json.dump({"object": out.get("object"), "units": units},
                  open(f"{d}/units.json", "w"), indent=1)
        print(f"[{os.path.basename(d.rstrip('/'))}] {out.get('object')}: "
              + "; ".join(f"{k}({len(v)}p)" for k, v in units.items()))


def legalize_unit(d, unit_name):
    from unit_legality import legalize
    u = json.load(open(f"{d}/units.json"))
    pids = u["units"][unit_name]
    meta = json.load(open(f"{d}/_src.json"))
    mesh = load_mesh(meta["glb"]); fids = np.load(meta["face_ids"])
    r = legalize([mesh], fids, pids, unit_name)
    print(f"'{unit_name}': {len(pids)} -> {len(r['final_pids'])} pids "
          f"(+sym {len(r['pulled_symmetry'])}, +floater {len(r['pulled_floaters'])}) "
          f"share {r['share']} -> {r['verdict']}")
    json.dump(r, open(f"{d}/legal_{unit_name.replace(' ','_')}.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "geom":
        glb, fid, out = sys.argv[2], sys.argv[3], sys.argv[4]
        photo = sys.argv[sys.argv.index("--photo") + 1] if "--photo" in sys.argv else None
        geom(glb, fid, out, photo)
        json.dump({"glb": os.path.abspath(glb), "face_ids": os.path.abspath(fid)},
                  open(f"{out}/_src.json", "w"))
    elif cmd == "name":
        name(sys.argv[2:])
    elif cmd == "legalize":
        legalize_unit(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
