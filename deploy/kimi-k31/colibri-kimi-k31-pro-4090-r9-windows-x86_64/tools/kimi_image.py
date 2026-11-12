"""Kimi K3 / K3.1 pictures for kimi_k3.c's vision tower (k3_vision.h).

The checkpoint's own processor (kimi_k3_vision_processing.py and media_utils.py, MoonViT's
media_proc_cfg in preprocessor_config.json), step for step:

  - the picture opened as it is (no RGB conversion yet: transparent_bg_fill_stage is
    "after_resize"), its size W x H kept for the prompt ("image WxH");
  - navit_resize_image: scaled down (never up) so that it has at most in_patch_limit
    (65,536) patches of 14 and at most patch_limit_on_one_side (512) a side, int() of the
    scaled sizes; then padded at the bottom and the right to multiples of 28 (patch x merge);
  - resized with Pillow's BICUBIC, then composited over the configured background if it has
    an alpha channel (a chessboard of 8-pixel squares, 255 and 180), else converted to RGB;
  - the padding is black (0).

The engine gets those pixels as RGB bytes (the IMAGE frame) and does the rest: it normalizes
them as media_utils.normalize does, cuts the 14 x 14 patches and runs the tower. A picture is
(gh / 2) x (gw / 2) rows of the language model's input, one <|media_pad|> each.

max_tokens (K3_MAX_IMAGE_TOKENS, 4,096 by default) caps a picture below the processor's own
16,384: the tower attends over all of a picture's patches, and 16,384 patches take about a
minute on the CPU. A picture over the cap is scaled down further, through the same
in_patch_limit the processor uses, until it fits.
"""
import io
import json
import math
from pathlib import Path

PATCH = 14
MERGE = 2
IN_PATCH_LIMIT = 65536
SIDE_LIMIT = 512
MAX_TOKENS = 4096


def _cfg(model_dir):
    """media_proc_cfg of the checkpoint's preprocessor_config.json, or the release's values."""
    cfg = {"patch_size": PATCH, "merge_kernel_size": MERGE, "in_patch_limit": IN_PATCH_LIMIT,
           "patch_limit_on_one_side": SIDE_LIMIT, "fixed_output_tokens": None,
           "transparent_bg_config": {"pattern": "chessboard", "chessboard_square_size": 8,
                                     "chessboard_square_on_top_left": True,
                                     "chessboard_white_value": 255, "chessboard_gray_value": 180},
           "transparent_bg_fill_stage": "after_resize"}
    if model_dir:
        try:
            raw = json.loads((Path(model_dir) / "preprocessor_config.json").read_text(encoding="utf-8"))
            given = raw.get("media_proc_cfg") if isinstance(raw, dict) else None
            if isinstance(given, dict):
                cfg.update({k: v for k, v in given.items() if k in cfg})
        except (OSError, ValueError):
            pass
    if cfg["patch_size"] != PATCH or cfg["merge_kernel_size"] != MERGE or cfg["fixed_output_tokens"] is not None:
        raise ValueError("this checkpoint's image processor is not the one kimi_k3.c's tower reads "
                         "(patch 14, merge 2, no fixed token count)")
    return cfg


def navit_resize(width, height, in_patch_limit, side_limit, patch=PATCH, merge=MERGE):
    """media_utils.navit_resize_image: (new_w, new_h, pad_w, pad_h, tokens)."""
    s1 = math.sqrt(in_patch_limit / (max(1.0, width // patch) * max(1.0, height // patch)))
    s2 = side_limit * patch / width
    s3 = side_limit * patch / height
    scale = min(1.0, s1, s2, s3)
    new_w, new_h = max(1, int(width * scale)), max(1, int(height * scale))
    new_w = min(new_w, side_limit * patch)
    new_h = min(new_h, side_limit * patch)
    factor = merge * patch
    pad_h = (factor - new_h % factor) % factor
    pad_w = (factor - new_w % factor) % factor
    tokens = ((new_h + pad_h) // factor) * ((new_w + pad_w) // factor)
    return new_w, new_h, pad_w, pad_h, tokens


def _chessboard(height, width, square, top_left, white, gray):
    import numpy as np
    bg = np.ones((height, width, 3), dtype=np.uint8) * white
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (y // square + x // square) % 2 == (1 if top_left else 0):
                bg[y:y + square, x:x + square] = gray
    return bg


def _fill_transparent(image, bg):
    """media_utils.fill_transparent_bg_with."""
    import numpy as np
    from PIL import Image
    if bg is None:
        return image.convert("RGB")
    if image.mode == "RGB":
        return image
    if not ("A" in image.getbands() or "transparency" in image.info):
        return image.convert("RGB")
    img = np.array(image.convert("RGBA"))
    height, width = img.shape[:2]
    pattern = bg.get("pattern", "black")
    if pattern == "white":
        back = np.full((height, width, 3), 255, dtype=np.uint8)
    elif pattern == "black":
        back = np.zeros((height, width, 3), dtype=np.uint8)
    elif pattern == "gray":
        back = np.full((height, width, 3), 128, dtype=np.uint8)
    elif pattern == "chessboard":
        back = _chessboard(height, width, bg.get("chessboard_square_size", 16),
                           bg.get("chessboard_square_on_top_left", True),
                           bg.get("chessboard_white_value", 255), bg.get("chessboard_gray_value", 200))
    else:
        raise ValueError(f"invalid background pattern {pattern!r}")
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    alpha = np.stack([alpha] * 3, axis=2)
    out = (alpha * img[:, :, :3] + (1 - alpha) * back).astype(np.uint8)
    return Image.fromarray(out)


def preprocess(data, model_dir=None, max_tokens=MAX_TOKENS):
    """Picture bytes -> (rgb uint8 [gh*14][gw*14][3], gh, gw, width, height, tokens)."""
    import numpy as np
    from PIL import Image
    cfg = _cfg(model_dir)
    image = Image.open(io.BytesIO(data))
    after = cfg.get("transparent_bg_fill_stage", "before_resize") == "after_resize"
    bg = cfg.get("transparent_bg_config")
    if not after:
        image = _fill_transparent(image, bg)
    width, height = image.size
    limit = int(cfg["in_patch_limit"])
    side = int(cfg["patch_limit_on_one_side"])
    new_w, new_h, pad_w, pad_h, tokens = navit_resize(width, height, limit, side)
    while max_tokens and tokens > max_tokens and limit > 4:
        limit = min(limit - 1, max(4, int(limit * max_tokens / tokens)))
        new_w, new_h, pad_w, pad_h, tokens = navit_resize(width, height, limit, side)
    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    if after:
        image = _fill_transparent(image, bg)
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        rgb = np.asarray(image.convert("RGB"))
    rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
    gh, gw = rgb.shape[0] // PATCH, rgb.shape[1] // PATCH
    return np.ascontiguousarray(rgb, dtype=np.uint8), gh, gw, width, height, tokens
