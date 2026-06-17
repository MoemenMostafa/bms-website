#!/usr/bin/env python3
"""Generate brand imagery for the BMS site via Hugging Face Inference API.

Reads HF_TOKEN from the environment. Writes PNGs into assets/generated/.
Usage: python3 gen_images.py [key1 key2 ...]   (no args = generate all)
"""
import os, sys, time, json, pathlib, urllib.request, urllib.error

TOKEN = os.environ["HF_TOKEN"]
OUT = pathlib.Path(__file__).resolve().parent.parent / "assets" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "black-forest-labs/FLUX.1-schnell"
ENDPOINT = f"https://router.huggingface.co/hf-inference/models/{MODEL}"

STYLE = ("premium editorial 3D render, brand palette crimson red #CC0000 and "
         "golden yellow #FFC200 on clean white, soft studio lighting, subtle depth "
         "of field, modern, sophisticated, no text, no words, no logos, no letters")

IMAGES = {
    "hero": (
        "Bold abstract dynamic composition for a creative marketing agency hero banner: "
        "flowing crimson-red and golden-yellow gradient ribbons, geometric energy, "
        "sense of momentum and ambition, " + STYLE, 1344, 768),
    "svc-strategy": (
        "Abstract concept of marketing strategy and planning: interlocking geometric "
        "pathways and a glowing target, chess-like precision, " + STYLE, 1024, 1024),
    "svc-branding": (
        "Abstract concept of brand identity and design: floating paint swatches, a bold "
        "abstract emblem, creative color burst, " + STYLE, 1024, 1024),
    "svc-performance": (
        "Abstract concept of performance marketing and growth: rising 3D bar and line "
        "charts, upward arrows, data flow, momentum, " + STYLE, 1024, 1024),
    "svc-social": (
        "Abstract concept of social media engagement: floating chat bubbles, heart and "
        "share icons rendered as sculptural 3D shapes, network connections, " + STYLE,
        1024, 1024),
    "about": (
        "Abstract creative agency atmosphere blending modern Saudi Vision 2030 ambition "
        "with German precision: sleek architectural lines, a sunrise of red and gold over "
        "a minimal cityscape skyline silhouette, optimistic, " + STYLE, 1344, 896),
    "cta": (
        "Wide abstract banner of converging crimson and gold light streaks on dark "
        "charcoal, premium, energetic, cinematic, " + STYLE, 1344, 600),
}


def generate(key, prompt, w, h, retries=6):
    body = json.dumps({
        "inputs": prompt,
        "parameters": {"width": w, "height": h},
    }).encode()
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {TOKEN}",
                     "Content-Type": "application/json",
                     "Accept": "image/png", "x-wait-for-model": "true"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = r.read()
            ctype = r.headers.get("Content-Type", "")
            if "image" not in ctype:
                print(f"  [{key}] unexpected content-type {ctype}: {data[:200]}")
                time.sleep(8); continue
            dest = OUT / f"{key}.png"
            dest.write_bytes(data)
            print(f"  [{key}] OK -> {dest} ({len(data)} bytes)")
            return True
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            print(f"  [{key}] HTTP {e.code} (attempt {attempt}): {msg}")
            if e.code in (503, 500, 429):
                time.sleep(12)
            else:
                return False
        except Exception as e:
            print(f"  [{key}] error (attempt {attempt}): {e}")
            time.sleep(10)
    return False


def main():
    keys = sys.argv[1:] or list(IMAGES.keys())
    ok = 0
    for k in keys:
        if k not in IMAGES:
            print(f"  [{k}] unknown key, skipping"); continue
        prompt, w, h = IMAGES[k]
        print(f"Generating {k} ({w}x{h})...")
        if generate(k, prompt, w, h):
            ok += 1
    print(f"Done: {ok}/{len(keys)} generated")
    sys.exit(0 if ok == len(keys) else 1)


if __name__ == "__main__":
    main()
