# Running P3-SAM at scale — patches we needed

P3-SAM itself is Tencent's (`Hunyuan3D-Part/P3-SAM`); we can't ship it. These
are the changes we had to make to run it over a large asset pool. Every one of
them cost us a debugging session.

## Correctness / robustness

**`P3-SAM/model.py` (~L18)** — sonata checkpoint root is hardcoded to `/root/sonata`:
```python
download_root='<your_workdir>/sonata_cache'
```

**`P3-SAM/demo/auto_mask.py`, the `__main__` directory loop** — the stock script
*discards results* when `--save_mid_res 0`, and dies on the first bad mesh:
- resume-skip an asset when `aabb.npy` or `error.txt` exists;
- wrap each asset in try/except and write `error.txt` on failure (numba raises on
  zero-size / degenerate meshes);
- explicitly save `aabb.npy`, **`face_ids.npy`**, `mesh_clean.glb` — the returns
  are dropped otherwise. `face_ids.npy` is the file everything downstream needs.

## Throughput (we measured 34 -> 6 assets/min collapse on heavy assets)

Three fixes brought it back to ~17/min, and later ~34/min with worker-mgmt fixes:

1. **connected-components was O(n²)** — rewrite the region-growing with a
   `collections.deque` frontier.
2. **decimate before segmenting** — cap at ~200k faces (`fast_simplification`);
   keep a KDTree onto the ORIGINAL mesh so face_ids map back at full resolution.
   Heavy assets were the entire tail of the runtime distribution.
3. **per-asset `SIGALRM` timeout (~75 s)** — catches the pathological loops that
   are not any single one of the above (e.g. O(parts²) blowups). Without a hard
   timeout one asset can stall a whole worker.

Note: if you re-clone P3-SAM you must re-apply all of these, and
`pip install fast_simplification` again.

## Environment pins that bit us

- `spconv-cu121==2.3.8` on sm_90 / H200 — the cu120 2.3.6 wheel SIGFPEs.
- X-Part (if you use it) does NOT read the HF cache: `export HY3DGEN_MODELS=<dir>`
  with `<dir>/tencent/Hunyuan3D-Part -> <hf_snapshot>` symlinked, plus
  `pytorch_lightning scikit-image fpsample pymeshlab==2023.12.post3 addict torch_cluster`.
