#!/usr/bin/env python3
"""Downloads SIPI image volumes and converts them to binary PGM/PPM.

The CUDA program reads only Netpbm files, so this script converts the SIPI
TIFF images: grayscale images become .pgm and color images become .ppm.

Usage:
  python3 scripts/prepare_data.py [--volumes textures misc aerials]
                                  [--raw_dir data/raw] [--out_dir data/input]
"""

import argparse
import pathlib
import tarfile
import urllib.request

from PIL import Image

SIPI_URL = "https://sipi.usc.edu/database/{volume}.tar.gz"


def download(volume: str, raw_dir: pathlib.Path) -> pathlib.Path:
    archive = raw_dir / f"{volume}.tar.gz"
    if not archive.exists():
        print(f"Downloading {volume} ...")
        urllib.request.urlretrieve(SIPI_URL.format(volume=volume), archive)
    return archive


def convert_archive(archive: pathlib.Path, volume: str,
                    out_dir: pathlib.Path) -> int:
    count = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.lower().endswith(
                    (".tif", ".tiff")):
                continue
            with tar.extractfile(member) as handle:
                image = Image.open(handle)
                image.load()
            if image.mode not in ("L", "RGB"):
                image = image.convert("RGB" if "A" in image.mode or
                                      image.mode in ("P", "CMYK") else "L")
            stem = pathlib.Path(member.name).stem
            suffix = ".pgm" if image.mode == "L" else ".ppm"
            image.save(out_dir / f"{volume}_{stem}{suffix}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volumes", nargs="+",
                        default=["textures", "misc", "aerials"])
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/input")
    args = parser.parse_args()

    raw_dir = pathlib.Path(args.raw_dir)
    out_dir = pathlib.Path(args.out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for volume in args.volumes:
        count = convert_archive(download(volume, raw_dir), volume, out_dir)
        print(f"{volume}: converted {count} images")
        total += count
    print(f"Done: {total} images in {out_dir}")


if __name__ == "__main__":
    main()
