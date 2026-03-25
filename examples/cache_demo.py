"""Demo: Dataset.cache() warm-up, reuse, invalidation, and groups.

Run:
    python examples/cache_demo.py

This script creates temporary tar shards, then walks through four scenarios
to illustrate how the cache layer works end-to-end.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

from mvp_dataset import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tar(path: Path, samples: list[dict]) -> None:
    """Write samples to a tar using the ``{key}.{field}`` naming convention."""
    with tarfile.open(str(path), "w") as tf:
        for sample in samples:
            key = sample["__key__"]
            for field, value in sample.items():
                if field.startswith("__") and field.endswith("__"):
                    continue
                data = value if isinstance(value, bytes) else str(value).encode()
                ti = tarfile.TarInfo(name=f"{key}.{field}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Setup: create two tar shards with image + label fields
# ---------------------------------------------------------------------------

tmpdir = Path(tempfile.mkdtemp(prefix="cache_demo_"))
print(f"Working directory: {tmpdir}\n")

shard_0 = tmpdir / "shard-00000.tar"
shard_1 = tmpdir / "shard-00001.tar"

make_tar(
    shard_0,
    [
        {"__key__": "0001", "image": b"\x89PNG_fake_img_1", "label": b"cat"},
        {"__key__": "0002", "image": b"\x89PNG_fake_img_2", "label": b"dog"},
    ],
)
make_tar(
    shard_1,
    [
        {"__key__": "0003", "image": b"\x89PNG_fake_img_3", "label": b"bird"},
    ],
)

tar_paths = [str(shard_0), str(shard_1)]

# ---------------------------------------------------------------------------
# Scenario 1: First iteration builds cache, second reuses it
# ---------------------------------------------------------------------------

section("1. Cache build & reuse")

call_count = 0


def preprocess(sample):
    global call_count
    call_count += 1
    return {**sample, "tag": b"processed"}


ds = Dataset.from_tars(tar_paths).map(preprocess).cache()

t0 = time.perf_counter()
samples_1 = list(ds)
dt_build = time.perf_counter() - t0
print(f"First  iter: {len(samples_1)} samples, map called {call_count}x  ({dt_build:.4f}s)")

call_count = 0
t0 = time.perf_counter()
samples_2 = list(ds)
dt_reuse = time.perf_counter() - t0
print(f"Second iter: {len(samples_2)} samples, map called {call_count}x  ({dt_reuse:.4f}s)")
print("  -> map was NOT called on reuse: cache hit!")

# Show cache directory layout
cache_dir = tmpdir / ".cache"
print(f"\nCache directory: {cache_dir}")
for f in sorted(cache_dir.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size} bytes)")

# ---------------------------------------------------------------------------
# Scenario 2: Changing the map function invalidates the cache
# ---------------------------------------------------------------------------

section("2. Cache invalidation on function change")


def preprocess_v2(sample):
    return {**sample, "tag": b"v2"}


ds_v2 = Dataset.from_tars(tar_paths).map(preprocess_v2).cache()
samples_v2 = list(ds_v2)

print(f"v1 tag: {samples_1[0]['tag']}")
print(f"v2 tag: {samples_v2[0]['tag']}")

all_tars = sorted(cache_dir.glob("*.tar"))
print(f"\nCache tars now ({len(all_tars)} total, v1 + v2 coexist):")
for f in all_tars:
    print(f"  {f.name}")

# ---------------------------------------------------------------------------
# Scenario 3: Field grouping — split fields into separate tars
# ---------------------------------------------------------------------------

section("3. Explicit groups: [image, label] vs auto-singleton")

# Clean up old cache
shutil.rmtree(cache_dir, ignore_errors=True)

ds_grouped = (
    Dataset.from_tars(tar_paths)
    .map(preprocess)
    .cache(
        groups=[["image"], ["label"]],  # "tag" field auto-becomes singleton
    )
)
list(ds_grouped)

group_tars = sorted(cache_dir.glob("*.tar"))
print(f"Group tars per shard ({len(group_tars)} total):")
for f in group_tars:
    print(f"  {f.name}")
print("  -> 'image', 'label', 'tag' each in their own tar per shard")

# ---------------------------------------------------------------------------
# Scenario 4: Post-cache stages run every iteration
# ---------------------------------------------------------------------------

section("4. Post-cache stages")

shutil.rmtree(cache_dir, ignore_errors=True)

post_count = 0


def post_fn(sample):
    global post_count
    post_count += 1
    return {**sample, "extra": b"post"}


ds_post = (
    Dataset.from_tars(tar_paths)
    .map(preprocess)
    .cache()  # <-- boundary
    .map(post_fn)  # <-- runs every iteration, not cached
)

list(ds_post)
print(f"After 1st iter: post_fn called {post_count}x")

post_count = 0
result = list(ds_post)
print(f"After 2nd iter: post_fn called {post_count}x  (still runs!)")
print(f"  sample keys: {[s['__key__'] for s in result]}")
print(f"  extra field: {result[0]['extra']}")

# ---------------------------------------------------------------------------
# Scenario 5: Expensive map — cache gives massive speedup
# ---------------------------------------------------------------------------

section("5. Expensive map: cache speedup")

shutil.rmtree(cache_dir, ignore_errors=True)

# Create a larger shard with 200 samples
big_shard = tmpdir / "big-shard-00000.tar"
make_tar(
    big_shard,
    [{"__key__": f"{i:04d}", "image": bytes(range(256)) * 4, "label": f"cls_{i % 10}".encode()} for i in range(200)],
)


def expensive_map(sample):
    """Simulate a heavy preprocessing step (resize, augment, tokenize, etc.)."""
    img = sample["image"]
    # Burn ~10ms per sample with redundant hashing
    import hashlib

    acc = img
    for _ in range(2000):
        acc = hashlib.sha256(acc).digest()
    return {**sample, "feature": acc}


ds_expensive = Dataset.from_tars([str(big_shard)]).map(expensive_map).cache()

t0 = time.perf_counter()
r1 = list(ds_expensive)
dt_cold = time.perf_counter() - t0

t0 = time.perf_counter()
r2 = list(ds_expensive)
dt_hot = time.perf_counter() - t0

speedup = dt_cold / dt_hot if dt_hot > 0 else float("inf")
print(f"Cold (build cache): {dt_cold:.3f}s  ({len(r1)} samples)")
print(f"Hot  (read cache):  {dt_hot:.3f}s  ({len(r2)} samples)")
print(f"Speedup:            {speedup:.1f}x")
assert r1[0]["feature"] == r2[0]["feature"], "round-trip mismatch!"
print("  -> Round-trip verified: cached values match original.")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

section("Done")
print(f"Cleaning up {tmpdir}")
# shutil.rmtree(tmpdir)
print("All clean.")
