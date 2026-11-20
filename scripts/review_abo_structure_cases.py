"""Independent three-view structural critique on CPU alongside the frozen GPU run."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from tactile3d.vision.local_vlm import configure_qwen_pixels, make_messages

CASES = ("B07H8VCM68_A1RKKUPIHCS9HS", "B07QHYD94M_ATVPDKIKX0DER")
PROMPT = """Inspect views 00, 08 and 20 in order. Compare two plausible internal
constructions consistent with the exterior. Cite specific visible boundaries by view.
Identify parts that can actually be counted. State whether internal void, wall
thickness and hidden support material are visually identifiable. State one observation
that could discriminate the hypotheses. Keep your answer under 110 words."""
SYSTEM = "Audit physical structure using only these views. Do not estimate mass."


def main(root, model_path, output):
    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    configure_qwen_pixels(processor, 448**2)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    ).eval()
    result = dict(model=model_path.name, device="cpu", dtype="bfloat16", prompt=PROMPT,
                  system=SYSTEM, views=[0, 8, 20], greedy=True, cases={})
    for scene in CASES:
        folder, = root.glob(f"val_*/observations/scenes/{scene}/images")
        paths = [folder / f"{scene}_{view:02d}.png" for view in result["views"]]
        images = []
        for path in paths:
            with Image.open(path) as source:
                images.append(source.copy())
        inputs = processor.apply_chat_template(make_messages(PROMPT, images, SYSTEM),
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        result["cases"][scene] = dict(image_sha256=[hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths], answer=processor.decode(tokens[0, inputs.input_ids.shape[1]:],
            skip_special_tokens=True))
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "model", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    main(args.root, args.model, args.output)
