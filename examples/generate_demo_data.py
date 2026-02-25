"""Generate small multimodal demo data under examples/demo_data.

Outputs:
- image shards: 4 tar files, each containing 8 PNG images
- depth shards: 4 tar files, each containing 8 PNG images
- samples.jsonl: references both modalities with tar:// URIs
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import zlib
from collections.abc import Callable
from pathlib import Path


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    length = len(data).to_bytes(4, "big")
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return length + chunk_type + data + crc.to_bytes(4, "big")


def encode_png_rgb(width: int, height: int, pixel_fn: Callable[[int, int], tuple[int, int, int]]) -> bytes:
    """Encode a tiny RGB PNG with no external dependencies."""

    rows = bytearray()
    for y in range(height):
        rows.append(0)  # filter type 0 (None)
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            rows.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    ihdr = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08"  # bit depth
        + b"\x02"  # color type: truecolor
        + b"\x00"  # compression
        + b"\x00"  # filter
        + b"\x00"  # interlace
    )
    idat = zlib.compress(bytes(rows), level=9)
    signature = b"\x89PNG\r\n\x1a\n"
    return signature + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", idat) + _png_chunk(b"IEND", b"")


def image_png(global_idx: int, width: int, height: int) -> bytes:
    def pixel(x: int, y: int) -> tuple[int, int, int]:
        r = (global_idx * 31 + x * 9) % 256
        g = (global_idx * 47 + y * 13) % 256
        b = (global_idx * 59 + x * 3 + y * 5) % 256
        return r, g, b

    return encode_png_rgb(width, height, pixel)


def depth_png(global_idx: int, width: int, height: int) -> bytes:
    def pixel(x: int, y: int) -> tuple[int, int, int]:
        v = (global_idx * 11 + x * 7 + y * 17) % 256
        return v, v, v

    return encode_png_rgb(width, height, pixel)


def write_member(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo image/depth tar shards and JSONL refs")
    parser.add_argument("--output-dir", default="examples/demo_data", help="Output root directory")
    parser.add_argument("--num-shards", type=int, default=4, help="Shards per modality")
    parser.add_argument("--samples-per-shard", type=int, default=8, help="Samples per shard")
    parser.add_argument("--width", type=int, default=32, help="PNG width")
    parser.add_argument("--height", type=int, default=32, help="PNG height")
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    image_dir = output_root / "image"
    depth_dir = output_root / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for shard_id in range(args.num_shards):
        image_shard = image_dir / f"shard_{shard_id:06d}.tar"
        depth_shard = depth_dir / f"shard_{shard_id:06d}.tar"

        with tarfile.open(image_shard, "w") as image_tf, tarfile.open(depth_shard, "w") as depth_tf:
            for item_id in range(args.samples_per_shard):
                global_idx = shard_id * args.samples_per_shard + item_id
                key = f"sample_{global_idx:05d}"
                image_member_name = f"{key}.image.png"
                depth_member_name = f"{key}.depth.png"

                image_payload = image_png(global_idx, args.width, args.height)
                depth_payload = depth_png(global_idx, args.width, args.height)
                write_member(image_tf, image_member_name, image_payload)
                write_member(depth_tf, depth_member_name, depth_payload)

                records.append(
                    {
                        "id": key,
                        "caption": f"demo sample {global_idx}",
                        "image": f"tar://{image_shard.as_posix()}#{image_member_name}",
                        "depth": f"tar://{depth_shard.as_posix()}#{depth_member_name}",
                    }
                )

    jsonl_path = output_root / "samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"generated {len(records)} samples, "
        f"{args.num_shards} image tars, {args.num_shards} depth tars, "
        f"jsonl={jsonl_path}"
    )


if __name__ == "__main__":
    main()
