#!/usr/bin/env python3
"""Builds PNG before/after comparisons from the GPU pipeline output.

For each selected image it writes results/previews/<stem>_before_after.png:
  input | equalized | blurred | edges   (pipeline with --save_intermediate)
  input | result                        (single operations)
Images are downscaled to a common height so the strips stay small.

Usage:
  python3 scripts/make_previews.py [--input_dir data/input]
      [--output_dir data/output] [--preview_dir results/previews]
      [--count 12] [--height 256]
"""

import argparse
import pathlib

from PIL import Image, ImageDraw

PIPELINE_STAGES = [("_1_equalized", "equalized"), ("_2_blurred", "blurred"),
                   ("_pipeline", "sobel edges")]
SINGLE_OPS = ["equalize", "blur", "sobel", "pipeline"]


def labeled(image: Image.Image, label: str, height: int) -> Image.Image:
    width = max(1, round(image.width * height / image.height))
    tile = image.convert("RGB").resize((width, height), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height + 20), "white")
    canvas.paste(tile, (0, 20))
    ImageDraw.Draw(canvas).text((4, 4), label, fill="black")
    return canvas


def stages_for(stem: str, output_dir: pathlib.Path):
    stages = [(output_dir / f"{stem}{suffix}.pgm", label)
              for suffix, label in PIPELINE_STAGES]
    if all(path.exists() for path, _ in stages):
        return stages
    for op in SINGLE_OPS:
        path = output_dir / f"{stem}_{op}.pgm"
        if path.exists():
            return [(path, op)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--output_dir", default="data/output")
    parser.add_argument("--preview_dir", default="results/previews")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    preview_dir = pathlib.Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in (".pgm", ".ppm"))
    # Spread the selection over the whole (sorted) dataset so every SIPI
    # volume is represented.
    step = max(1, len(inputs) // max(1, args.count))
    written = 0
    for source in inputs[::step][:args.count]:
        stages = stages_for(source.stem, output_dir)
        if not stages:
            continue
        tiles = [labeled(Image.open(source), "input", args.height)]
        tiles += [labeled(Image.open(path), label, args.height)
                  for path, label in stages]
        strip = Image.new("RGB", (sum(t.width for t in tiles) +
                                  4 * (len(tiles) - 1), tiles[0].height),
                          "white")
        x = 0
        for tile in tiles:
            strip.paste(tile, (x, 0))
            x += tile.width + 4
        strip.save(preview_dir / f"{source.stem}_before_after.png")
        written += 1
    print(f"Wrote {written} previews to {preview_dir}")


if __name__ == "__main__":
    main()
