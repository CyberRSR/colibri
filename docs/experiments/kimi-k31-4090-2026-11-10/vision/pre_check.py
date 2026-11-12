import io, os, sys, types
import numpy as np
from PIL import Image
sys.modules["torchaudio"] = None
fake = types.ModuleType("transformers.models.llava.modeling_llava"); fake.LlavaCausalLMOutputWithPast = object
sys.modules["transformers.models.llava.modeling_llava"] = fake
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import kref.kimi_k3_vision_processing as P
sys.path.insert(0, r"C:/colibri/dev/.claude/worktrees/kimi-k3-main/c/tools")
import kimi_image
K31 = r"C:/AI/Kimi-K3.1-Pro"
pre = json.load(open(os.path.join(K31, "preprocessor_config.json")))
proc = P.KimiK3VisionProcessor(media_proc_cfg=pre["media_proc_cfg"])
def ref(data):
    m = P.ensure_media_type({"type": "image", "image": Image.open(io.BytesIO(data))}, proc._transparent_bg_config, proc._transparent_bg_fill_stage)
    rc = proc.get_resize_config(m)
    u8 = proc.resize_image(m["image"], rc["new_width"], rc["new_height"], rc["pad_width"], rc["pad_height"])
    return u8, rc["num_tokens"], m["image"].size
rng = np.random.default_rng(1)
cases = []
for path in ["small/shot.png", "big/shot.png"]:
    cases.append((path, open(path, "rb").read()))
def enc(img, fmt="PNG"):
    b = io.BytesIO(); img.save(b, fmt); return b.getvalue()
rgba = Image.fromarray(rng.integers(0, 256, (300, 517, 4), dtype=np.uint8), "RGBA")
cases.append(("rgba", enc(rgba)))
cases.append(("P+transp", enc(Image.open("small/shot.png").convert("P").copy())))
pimg = Image.open("small/shot.png").convert("P"); pimg.info["transparency"] = 3
cases.append(("P+trns", enc(pimg)))
cases.append(("L", enc(Image.open("big/shot.png").convert("L"))))
cases.append(("jpeg", enc(Image.open("big/shot.png"), "JPEG")))
cases.append(("huge", enc(Image.fromarray(rng.integers(0, 256, (4000, 6000, 3), dtype=np.uint8)))))
cases.append(("tiny", enc(Image.new("RGB", (5, 3), (10, 200, 30)))))
for name, data in cases:
    u8, nt, size = ref(data)
    rgb, gh, gw, w, h, tokens = kimi_image.preprocess(data, K31, max_tokens=None)
    same = rgb.shape == u8.shape and np.array_equal(rgb, u8)
    print(f"{name:10s} ref {u8.shape} {nt} tok, ours {rgb.shape} {tokens} tok size {(w,h)}=={size}: {'SAME' if same else 'DIFF'}")
rgb, gh, gw, w, h, tokens = kimi_image.preprocess(cases[-2][1], K31)
print("huge capped:", rgb.shape, gh, gw, tokens)
