"""Step 2 input prep: paint each geometry cluster a flat colour and build the
panel image the VLM will look at (+ a legend it can refer to by name).

Why colours and not "part 17": a VLM cannot address an integer id it cannot
see. Painting clusters and giving a legend turns grouping+naming into ONE
question it can actually answer, and the colour->cluster map makes the answer
machine-consumable with zero ambiguity.

Two renderers:
  render="points"  (default) zero extra deps — visible-only z-buffer point
                   splat, 3 views. Good enough for the VLM in our experience.
  render=<callable>(glb_path, out_dir) -> [image paths]
                   plug your own (Blender/pyrender/...) for solid shaded views.
"""
import os, json
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

PALETTE = [("red", (220, 40, 40)), ("blue", (40, 80, 230)), ("green", (40, 180, 60)),
           ("yellow", (235, 200, 30)), ("purple", (150, 60, 200)), ("orange", (240, 130, 30)),
           ("cyan", (40, 200, 210)), ("magenta", (230, 60, 180)), ("brown", (140, 90, 40)),
           ("pink", (250, 170, 190)), ("olive", (128, 128, 0)), ("navy", (20, 30, 120))]
_PAL = dict(PALETTE)


def effective_clusters(tree, cluster):
    """Colour by CO-MOVE unit, not raw cluster: a contained/attached child is
    painted with its parent so the VLM sees one thing, not confetti."""
    eff = {int(p): int(c) for p, c in cluster.items()}
    for pid, v in tree.items():
        pid = int(pid)
        if v["parent"] != -1 and eff.get(pid) == pid:
            eff[pid] = eff.get(int(v["parent"]), pid)
    return eff


def _splat(mesh, fids, eff, cmap, out, n=60000, res=420):
    """visible-only point render, 3 orthographic views (no GPU, no Blender)."""
    pts, fi = trimesh.sample.sample_surface(mesh, n)
    pid = fids[fi]
    col = np.full((len(pts), 3), 0.62)
    for p, c in eff.items():
        if c in cmap:
            col[pid == p] = np.array(_PAL[cmap[c][0]]) / 255.0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = pts.min(0), pts.max(0)
    q = ((pts - lo) / np.maximum(hi - lo, 1e-9) * (res - 1)).astype(int)
    outs = []
    for ax, u, v, nm in [(1, 0, 2, "front"), (0, 1, 2, "side"), (2, 0, 1, "top")]:
        depth = pts[:, ax] * (1 if nm == "top" else -1)
        order = np.argsort(-depth)
        keep = order[np.unique((q[:, u] * res + q[:, v])[order], return_index=True)[1]]
        f, a = plt.subplots(figsize=(4, 4))
        a.scatter(pts[keep][:, u], pts[keep][:, v], c=col[keep], s=2.2)
        a.set_aspect("equal"); a.axis("off")
        p = f"{out}/view_{nm}.png"
        f.savefig(p, dpi=105, bbox_inches="tight", facecolor="white"); plt.close(f)
        outs.append(p)
    return outs


def build_panel(mesh, face_ids, part_tree_out, out_dir, extra_image=None,
                render="points", max_colors=len(PALETTE)):
    """mesh: trimesh.Trimesh, face_ids: (F,), part_tree_out: build_part_tree()'s dict.
    Writes colored.glb, colors.json, panel.jpg. Returns colors dict."""
    os.makedirs(out_dir, exist_ok=True)
    fids = np.asarray(face_ids)
    eff = effective_clusters(part_tree_out["tree"], part_tree_out["cluster"])
    areas = mesh.area_faces
    ca = {}
    for pid, cl in eff.items():
        ca[cl] = ca.get(cl, 0.0) + float(areas[fids == pid].sum())
    ranked = sorted(ca, key=lambda c: -ca[c])[:max_colors]
    cmap = {cl: PALETTE[i] for i, cl in enumerate(ranked)}
    tot = sum(ca.values()) or 1.0

    sc = trimesh.Scene()
    colors = {}
    for cl in ranked:
        pids = [p for p, c in eff.items() if c == cl]
        mm = mesh.copy(); mm.update_faces(np.isin(fids, pids)); mm.remove_unreferenced_vertices()
        if not len(mm.faces):
            continue
        name, rgb = cmap[cl]
        mm.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[*rgb, 255], metallicFactor=0.0, roughnessFactor=0.9))
        sc.add_geometry(mm, geom_name=f"c_{name}")
        colors[name] = {"cluster": int(cl), "pids": sorted(int(p) for p in pids),
                        "area_share": round(ca[cl] / tot, 3)}
    rest = [p for p, c in eff.items() if c not in ranked]
    if rest:
        mm = mesh.copy(); mm.update_faces(np.isin(fids, rest)); mm.remove_unreferenced_vertices()
        if len(mm.faces):
            mm.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial(
                    baseColorFactor=[128, 128, 128, 255], metallicFactor=0.0, roughnessFactor=0.9))
            sc.add_geometry(mm, geom_name="c_gray")
    glb = f"{out_dir}/colored.glb"
    sc.export(glb)
    json.dump(colors, open(f"{out_dir}/colors.json", "w"), indent=1)

    views = (_splat(mesh, fids, eff, cmap, out_dir) if render == "points"
             else render(glb, out_dir))

    def sq(p, s=400):
        im = Image.open(p)
        if im.mode == "RGBA":
            im = Image.alpha_composite(Image.new("RGBA", im.size, (245,) * 3 + (255,)), im)
        return im.convert("RGB").resize((s, s), Image.LANCZOS)
    tiles = ([sq(extra_image)] if extra_image and os.path.exists(extra_image) else []) \
            + [sq(v) for v in views]
    LEG = 56
    panel = Image.new("RGB", (400 * len(tiles), 400 + LEG), (250, 250, 250))
    for i, t in enumerate(tiles):
        panel.paste(t, (i * 400, 0))
    dr = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    x = 10
    for name in colors:
        dr.rectangle([x, 412, x + 30, 442], fill=_PAL[name], outline=(0, 0, 0))
        dr.text((x + 36, 414), name, fill=(0, 0, 0), font=font)
        x += 40 + 12 * len(name) + 24
    panel.save(f"{out_dir}/panel.jpg", quality=92)
    return colors
