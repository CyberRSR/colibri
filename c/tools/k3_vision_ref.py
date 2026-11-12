#!/usr/bin/env python3
"""Kimi K3.1's vision tower run by Moonshot's own code: the oracle of k3_vision.h and k3cu_vision.

Maintainer-only (torch; the checkpoint's weights). modeling_kimi_k3.py ships with the Kimi-K3
release (moonshotai/Kimi-K3), not with K3.1, whose vision_config, media_utils.py and
kimi_k3_vision_processing.py are the same; the tower and the projector are imported from it
as they are, with two changes that do not touch a number: the varlen flash attention is the
file's own eager attention (one picture), and get_rope_shape runs without torch.compile.

  k3_vision_ref.py <picture> <out_dir> [--k31 DIR] [--k3 DIR] [--dtype f32|bf16] [--eps-bf16]

Writes into out_dir: pixels.u8 (the processor's resized and padded picture, what the gateway
sends the engine), grid.txt ("grid_h grid_w width height tokens"), and from the f32 run
pe.f32, b0.f32, enc.f32 (after the patches and positions, after block 0, after the final norm)
and rows.f32 [tokens][7168] (the projector's), rows_bf16.f32 from --dtype bf16. Then:

  tools/k3_vision_check <K31 dir> <out_dir>/pixels.u8 <grid_h> <grid_w> <c_dir> [gpu]

and compare. nn.RMSNorm(hidden) has no eps in this code: torch takes it from the type it
computes in. --eps-bf16 sets bf16's own 2^-7 instead, to see what that would give (rows 55% away
from the bf16 run; FLT_EPSILON's are 4% away, bf16's noise).
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import types


def load_moonshot(k31, k3, work):
    """Moonshot's processor and modeling code as a package in `work`."""
    pkg = os.path.join(work, "kref")
    os.makedirs(pkg, exist_ok=True)
    for src, name in [(k31, "media_utils.py"), (k31, "kimi_k3_vision_processing.py"),
                      (k3, "modeling_kimi_k3.py"), (k31, "configuration_kimi_k3.py")]:
        shutil.copyfile(os.path.join(src, name), os.path.join(pkg, name))
    with open(os.path.join(pkg, "modeling_kimi_linear.py"), "w") as f:
        f.write("class KimiLinearForCausalLM:\n    pass\n")   # the language model is not run here
    open(os.path.join(pkg, "__init__.py"), "w").close()
    # transformers' llava module wants only an output class here; torchaudio, if it is built
    # for another CUDA than torch, would stop transformers' import: it is not needed
    sys.modules.setdefault("torchaudio", None)
    fake = types.ModuleType("transformers.models.llava.modeling_llava")
    fake.LlavaCausalLMOutputWithPast = object
    sys.modules["transformers.models.llava.modeling_llava"] = fake
    sys.path.insert(0, work)
    import kref.modeling_kimi_k3 as M
    import kref.kimi_k3_vision_processing as P
    import torch.nn.functional as F

    def get_rope_shape(org, interpolation_mode, shape):   # the same, without torch.compile
        return (F.interpolate(org.permute((2, 0, 1)).unsqueeze(0), size=shape, mode=interpolation_mode)
                .squeeze(0).permute((1, 2, 0)).flatten(end_dim=1))
    M.get_rope_shape = get_rope_shape
    return M, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("picture")
    ap.add_argument("out")
    ap.add_argument("--k31", default=r"C:/AI/Kimi-K3.1-Pro")
    ap.add_argument("--k3", default=r"C:/AI/Kimi-K3", help="a folder holding modeling_kimi_k3.py")
    ap.add_argument("--dtype", default="f32", choices=["f32", "bf16"])
    ap.add_argument("--eps-bf16", action="store_true")
    a = ap.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    work = tempfile.mkdtemp(prefix="k3vref_")
    M, P = load_moonshot(a.k31, a.k3, work)
    cfg = json.load(open(os.path.join(a.k31, "config.json")))
    vc = types.SimpleNamespace(**cfg["vision_config"])
    pre = json.load(open(os.path.join(a.k31, "preprocessor_config.json")))
    proc = P.KimiK3VisionProcessor(media_proc_cfg=pre["media_proc_cfg"])
    feat = proc.preprocess([{"type": "image", "image": Image.open(a.picture)}], return_tensors="pt")
    pv, grid = feat["pixel_values"], feat["grid_thws"]
    _t, gh, gw = [int(x) for x in grid[0]]
    media = P.ensure_media_type({"type": "image", "image": Image.open(a.picture)}, proc._transparent_bg_config,
                                proc._transparent_bg_fill_stage)
    rc = proc.get_resize_config(media)
    u8 = proc.resize_image(media["image"], rc["new_width"], rc["new_height"], rc["pad_width"], rc["pad_height"])
    os.makedirs(a.out, exist_ok=True)
    np.ascontiguousarray(u8, dtype=np.uint8).tofile(os.path.join(a.out, "pixels.u8"))
    w0, h0 = Image.open(a.picture).size
    with open(os.path.join(a.out, "grid.txt"), "w") as f:
        f.write(f"{gh} {gw} {w0} {h0} {rc['num_tokens']}\n")
    print(f"picture {w0}x{h0}: grid {gh}x{gw}, {rc['num_tokens']} rows")

    vt_cfg = M.VisionTowerConfig(vc)
    vt_cfg._attn_implementation = "eager"
    tower = M.MoonViT3dPretrainedModel(vt_cfg)
    proj = M.PatchMergerMLPV2(M.ProjectorConfig(vc))
    wmap = json.load(open(os.path.join(a.k31, "model.safetensors.index.json")))["weight_map"]
    by_file = {}
    for name, fn in wmap.items():
        if name.startswith(("vision_tower.", "mm_projector.")):
            by_file.setdefault(fn, []).append(name)
    st_t, st_p = {}, {}
    for fn, names in by_file.items():
        with safe_open(os.path.join(a.k31, fn), "pt") as f:
            for n in names:
                (st_t if n.startswith("vision_tower.") else st_p)[n.split(".", 1)[1]] = f.get_tensor(n)
    print("tower:", tower.load_state_dict(st_t, strict=True), "projector:", proj.load_state_dict(st_p, strict=True))
    if a.eps_bf16:
        for mod in tower.modules():
            if isinstance(mod, torch.nn.RMSNorm) and mod.eps is None:
                mod.eps = 2.0 ** -7
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if a.dtype == "f32" else torch.bfloat16
    tower, proj = tower.to(dev, dt).eval(), proj.to(dev, dt).eval()
    keep = {}
    tower.patch_embed.register_forward_hook(lambda m, i, o: keep.__setitem__("pe", o.float().cpu()))
    tower.encoder.blocks[0].register_forward_hook(lambda m, i, o: keep.__setitem__("b0", o.float().cpu()))
    tower.encoder.final_layernorm.register_forward_hook(lambda m, i, o: keep.__setitem__("enc", o.float().cpu()))
    with torch.no_grad():
        rows = proj(tower(pv.to(dev, dt), grid.to(dev)))[0].float().cpu().numpy()
    tag = ("" if a.dtype == "f32" else "_bf16") + ("_eps7" if a.eps_bf16 else "")
    rows.tofile(os.path.join(a.out, f"rows{tag}.f32"))
    if not tag:
        for k, v in keep.items():
            v.numpy().astype(np.float32).tofile(os.path.join(a.out, f"{k}.f32"))
    print(f"rows {rows.shape} -> {os.path.join(a.out, f'rows{tag}.f32')}")
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
