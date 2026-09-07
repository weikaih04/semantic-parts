"""Step 3: ONE VLM call that GROUPS AND NAMES at the same time.

Input  : the colour panel + the legend colour list
Output : {"object": str, "groups": [{"name": str, "colors": [str]}]}
         -> resolve_groups() turns that into {name: [pid,...]}

Design notes worth keeping if you reimplement:
  * grouping and naming in ONE call. Splitting them makes the model name things
    it has not committed to grouping, and the two answers drift.
  * the model answers in COLOUR NAMES, never part indices — it can see colours.
    The colour->pids map (colors.json) does the machine-side resolution, so the
    answer is unambiguous by construction.
  * guided/structured JSON decoding (vLLM StructuredOutputsParams). Free-form
    JSON from a 27B model fails to parse often enough to matter at scale.
  * show a REAL RENDER next to the painted views when you have one. Grouping is
    a semantic judgement; the model needs to know it is a dragon, not just a set
    of blobs.
  * we render painted views, NOT raw point scatter with an index legend — an
    earlier version asked about scatter plots and mislabelled a jaw as
    "backrest". Paint + legend fixed it.

Deps: vllm (tested on 0.23), a VL model. We use Qwen3.6-27B.
"""
import json
from PIL import Image

SCHEMA = {
    "type": "object",
    "properties": {
        "object": {"type": "string", "maxLength": 60},
        "groups": {"type": "array", "maxItems": 12, "items": {
            "type": "object",
            "properties": {"name": {"type": "string", "maxLength": 40},
                           "colors": {"type": "array", "maxItems": 12,
                                      "items": {"type": "string", "maxLength": 12}}},
            "required": ["name", "colors"]}}},
    "required": ["object", "groups"],
}

PROMPT_TMPL = (
    "The leftmost panel is a photo of a 3D object. The other panels show the SAME object with its "
    "geometry clusters painted in flat colors (legend at the bottom; gray = small leftover pieces).\n"
    "Available colors: {colors}.\n"
    "Task: group the colors into SEMANTIC PARTS a person would name when editing this object "
    "(e.g. head, body, hat, weapon, base). Rules:\n"
    "- every listed color appears in exactly one group;\n"
    "- colors that belong to the same nameable thing go together (e.g. if the head is painted in "
    "two colors, both go in the 'head' group);\n"
    "- name each group in 1-3 lowercase words specific to THIS object;\n"
    "- also name the whole object.\n"
    "Output ONLY JSON. /no_think")


def build_llm(model="Qwen/Qwen3.6-27B", tp=1, gpu_util=0.85, max_seqs=16):
    from vllm import LLM
    return LLM(model=model, max_model_len=8192, gpu_memory_utilization=gpu_util,
               tensor_parallel_size=tp, limit_mm_per_prompt={"image": 1},
               dtype="bfloat16", max_num_seqs=max_seqs)


def ask(llm, items, max_tokens=512):
    """items: [(panel_jpg_path, colors_dict)] -> [parsed dict]  (batched)."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
    batch = []
    for panel, colors in items:
        im = Image.open(panel).convert("RGB")
        im.thumbnail((2000, 640))
        batch.append({"prompt": ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                                 + PROMPT_TMPL.format(colors=", ".join(colors))
                                 + "<|im_end|>\n<|im_start|>assistant\n"),
                      "multi_modal_data": {"image": im}})
    outs = llm.generate(batch, sp)
    res = []
    for o in outs:
        t = o.outputs[0].text.strip()
        try:
            res.append(json.loads(t[t.index("{"):t.rindex("}") + 1]))
        except Exception:
            res.append({"object": "?", "groups": [], "raw": t[:300]})
    return res


def resolve_groups(vlm_out, colors):
    """{'name': [pid,...]} from the VLM answer + the colour->pids map."""
    out = {}
    for g in vlm_out.get("groups", []):
        pids = []
        for c in g.get("colors", []):
            c = c.strip().lower()
            if c in colors:
                pids += colors[c]["pids"]
        if pids:
            out[g["name"]] = sorted(set(pids))
    return out
