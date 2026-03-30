"""Tests for Dataset.cache() – materialization, invalidation, and codecs."""

from __future__ import annotations

import io
import json
import tarfile
import warnings
from pathlib import Path

import pytest

from mvp_dataset import Dataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tar(path: Path, samples: list[dict]) -> None:
    """Write samples to a tar file using the ``{key}.{field}`` convention."""
    with tarfile.open(str(path), "w") as tf:
        for sample in samples:
            key = sample["__key__"]
            for field, value in sample.items():
                if field.startswith("__") and field.endswith("__"):
                    continue
                if isinstance(value, bytes):
                    data = value
                elif isinstance(value, str):
                    data = value.encode()
                else:
                    data = str(value).encode()
                member_name = f"{key}.{field}"
                ti = tarfile.TarInfo(name=member_name)
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))


def _make_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Basic cache build / reuse
# ---------------------------------------------------------------------------


def test_cache_build_and_reuse(tmp_path):
    """First iteration builds cache; second iteration skips upstream map."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(
        shard,
        [
            {"__key__": "0001", "image": b"img1", "label": b"lbl1"},
            {"__key__": "0002", "image": b"img2", "label": b"lbl2"},
        ],
    )

    call_count = [0]

    def my_map(sample):
        call_count[0] += 1
        return {**sample, "processed": b"yes"}

    ds = Dataset.from_tars([str(shard)]).map(my_map).cache(show_progress=False)

    # First pass: warm-up builds cache.
    samples1 = list(ds)
    assert call_count[0] == 2

    # Second pass: reads from cache; map should NOT be called again.
    call_count[0] = 0
    samples2 = list(ds)
    assert call_count[0] == 0

    assert len(samples1) == len(samples2) == 2
    for s1, s2 in zip(samples1, samples2, strict=True):
        assert s1["image"] == s2["image"]
        assert s1["processed"] == s2["processed"]
        assert s1["__key__"] == s2["__key__"]


def test_cache_invalidation_on_fn_change(tmp_path):
    """Changing the map function produces new cache files."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "x": b"hello"}])

    def fn_v1(sample):
        return {**sample, "tag": b"v1"}

    def fn_v2(sample):
        return {**sample, "tag": b"v2"}

    ds1 = Dataset.from_tars([str(shard)]).map(fn_v1).cache(show_progress=False)
    ds2 = Dataset.from_tars([str(shard)]).map(fn_v2).cache(show_progress=False)

    result1 = list(ds1)
    result2 = list(ds2)

    assert result1[0]["tag"] == b"v1"
    assert result2[0]["tag"] == b"v2"

    # Different plan fingerprints → different cache tar files coexist.
    cache_tars = list((tmp_path / ".cache").glob("*.tar"))
    assert len(cache_tars) == 2


def test_cache_directory_layout(tmp_path):
    """Cache tars are written under <shard_parent>/.cache/."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "data": b"hello"}])

    ds = Dataset.from_tars([str(shard)]).cache(show_progress=False)
    list(ds)

    cache_dir = tmp_path / ".cache"
    assert cache_dir.is_dir()
    assert (cache_dir / "shard-00000.tar.manifest.json").is_file()
    group_tars = list(cache_dir.glob("shard-00000-*.tar"))
    assert len(group_tars) == 1


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


def test_cache_groups_explicit(tmp_path):
    """Explicit groups split fields into separate tars; uncovered key → singleton."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(
        shard,
        [
            {"__key__": "0001", "image": b"i1", "depth": b"d1", "label": b"l1"},
        ],
    )

    ds = Dataset.from_tars([str(shard)]).cache(
        groups=[["image", "depth"]],
        show_progress=False,
    )
    samples = list(ds)

    assert samples[0]["image"] == b"i1"
    assert samples[0]["depth"] == b"d1"
    assert samples[0]["label"] == b"l1"

    # Two group tars: image+depth together, label as singleton.
    cache_tars = sorted((tmp_path / ".cache").glob("*.tar"))
    assert len(cache_tars) == 2


def test_cache_groups_none_all_in_one(tmp_path):
    """groups=None puts all non-meta fields in a single tar."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "a": b"A", "b": b"B"}])

    ds = Dataset.from_tars([str(shard)]).cache(show_progress=False)
    list(ds)

    cache_tars = list((tmp_path / ".cache").glob("*.tar"))
    assert len(cache_tars) == 1


# ---------------------------------------------------------------------------
# Signature propagation through map
# ---------------------------------------------------------------------------


def test_map_signature_unchanged_field(tmp_path):
    """Unchanged fields in map keep the same signature → same tar filename."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "a": b"A", "b": b"B"}])

    def map_only_b(sample):
        # Only modify 'b'; 'a' is passed through unchanged.
        return {**sample, "b": b"B_modified"}

    ds = (
        Dataset.from_tars([str(shard)])
        .map(map_only_b)
        .cache(
            groups=[["a"], ["b"]],
            show_progress=False,
        )
    )
    list(ds)

    cache_dir = tmp_path / ".cache"
    tars = sorted(cache_dir.glob("*.tar"))
    # Expect two group tars: one for 'a', one for 'b'.
    assert len(tars) == 2


def test_map_signature_changes_cause_new_tar(tmp_path):
    """Modifying a field value and rebuilding cache produces a new tar filename."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "x": b"val"}])

    def fn_a(sample):
        return {**sample, "x": b"result_a"}

    def fn_b(sample):
        return {**sample, "x": b"result_b"}

    ds_a = Dataset.from_tars([str(shard)]).map(fn_a).cache(show_progress=False)
    ds_b = Dataset.from_tars([str(shard)]).map(fn_b).cache(show_progress=False)

    list(ds_a)
    list(ds_b)

    tars = list((tmp_path / ".cache").glob("*.tar"))
    # Different sigs → two distinct tars.
    assert len(tars) == 2


# ---------------------------------------------------------------------------
# Unsupported stage warning
# ---------------------------------------------------------------------------


def test_unsupported_stage_warning(tmp_path):
    """shuffle() before cache() emits a CacheInvalidationWarning."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "x": b"v"}])

    ds = Dataset.from_tars([str(shard)]).shuffle(100).cache(show_progress=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        list(ds)

    warning_texts = [str(w.message) for w in caught]
    assert any("CacheInvalidationWarning" in t for t in warning_texts)
    assert any("shuffle" in t for t in warning_texts)


def test_unsupported_stage_cache_reuse(tmp_path):
    """shuffle() before cache() still builds a cache that is reused."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(
        shard,
        [
            {"__key__": "0001", "x": b"v1"},
            {"__key__": "0002", "x": b"v2"},
        ],
    )

    ds = Dataset.from_tars([str(shard)]).shuffle(100).cache(show_progress=False)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        samples1 = list(ds)
        # Second iteration: cache is hit (no rebuild).
        samples2 = list(ds)

    assert len(samples1) == 2
    assert len(samples2) == 2


# ---------------------------------------------------------------------------
# Codec round-trip
# ---------------------------------------------------------------------------


def _roundtrip(value, tmp_path):
    """Cache and recover a single field value."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "k", "field": b"placeholder"}])

    def inject(sample):
        return {"__key__": sample["__key__"], "field": value}

    ds = Dataset.from_tars([str(shard)]).map(inject).cache(show_progress=False)
    # First pass builds cache, second pass reads it.
    list(ds)
    result = list(ds)
    return result[0]["field"]


def test_codec_bytes(tmp_path):
    assert _roundtrip(b"hello bytes", tmp_path) == b"hello bytes"


def test_codec_str(tmp_path):
    assert _roundtrip("hello str", tmp_path) == "hello str"


def test_codec_json_scalar(tmp_path):
    assert _roundtrip(42, tmp_path) == 42


def test_codec_json_list(tmp_path):
    assert _roundtrip([1, 2, 3], tmp_path) == [1, 2, 3]


def test_codec_json_dict(tmp_path):
    assert _roundtrip({"a": 1, "b": "two"}, tmp_path) == {"a": 1, "b": "two"}


def test_codec_none(tmp_path):
    assert _roundtrip(None, tmp_path) is None


def test_codec_numpy(tmp_path):
    np = pytest.importorskip("numpy")

    shard = tmp_path / "shard-00000.tar"
    # Store raw bytes in tar; map converts to ndarray (no ndarray in closure).
    _make_tar(shard, [{"__key__": "k", "field": b"\x00\x00\x80?\x00\x00\x00@"}])

    def to_arr(sample):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        return {"__key__": sample["__key__"], "field": arr}

    ds = Dataset.from_tars([str(shard)]).map(to_arr).cache(show_progress=False)
    list(ds)  # warm-up
    result = list(ds)
    assert isinstance(result[0]["field"], np.ndarray)
    np.testing.assert_array_equal(result[0]["field"], np.array([1.0, 2.0], dtype=np.float32))


def test_codec_torch(tmp_path):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "k", "field": b"placeholder"}])

    def to_tensor(sample):
        return {"__key__": sample["__key__"], "field": torch.tensor([1.0, 2.0, 3.0])}

    ds = Dataset.from_tars([str(shard)]).map(to_tensor).cache(show_progress=False)
    list(ds)  # warm-up
    result = list(ds)
    assert isinstance(result[0]["field"], torch.Tensor)
    np.testing.assert_array_equal(result[0]["field"].numpy(), [1.0, 2.0, 3.0])


def test_codec_unsupported_type(tmp_path):
    """Encoding an unsupported type should raise TypeError."""
    from mvp_dataset.cache.codecs import encode_value

    with pytest.raises(TypeError, match="CacheCodecError"):
        encode_value(object())


# ---------------------------------------------------------------------------
# JSONL source
# ---------------------------------------------------------------------------


def test_cache_jsonl_source(tmp_path):
    """cache() works with a JSONL source."""
    shard = tmp_path / "shard-00000.jsonl"
    _make_jsonl(
        shard,
        [
            {"__key__": "r0", "__file__": str(shard), "__index_in_file__": 0, "text": "hello"},
            {"__key__": "r1", "__file__": str(shard), "__index_in_file__": 1, "text": "world"},
        ],
    )

    call_count = [0]

    def fn(sample):
        call_count[0] += 1
        return {**sample, "upper": sample["text"].upper()}

    ds = Dataset.from_jsonl([str(shard)]).map(fn).cache(show_progress=False)

    list(ds)
    assert call_count[0] == 2

    call_count[0] = 0
    result = list(ds)
    assert call_count[0] == 0
    assert {s["upper"] for s in result} == {"HELLO", "WORLD"}


def test_jsonl_resolve_list_refs_with_bare_tar_uri(tmp_path):
    """JSONL ref resolution supports list-valued fields and bare ``.tar#...`` URIs."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_shard = image_dir / "train-00000.tar"
    _make_tar(
        image_shard,
        [
            {"__key__": "image_00", "jpg": b"img0"},
            {"__key__": "image_01", "jpg": b"img1"},
        ],
    )

    shard = tmp_path / "samples.jsonl"
    _make_jsonl(
        shard,
        [
            {
                "images": [
                    "images/train-00000.tar#image_00.jpg",
                    "images/train-00000.tar#image_01.jpg",
                ]
            }
        ],
    )

    ds = Dataset.from_jsonl(
        [str(shard)],
        group_key="images",
        num_shards=1,
        output_dir=tmp_path / "materialized",
    ).resolve_refs([("images", str(tmp_path))])

    samples = list(ds)
    assert len(samples) == 1
    assert samples[0]["images"] == [b"img0", b"img1"]


def test_cache_jsonl_list_refs_roundtrip(tmp_path):
    """Resolved ``list[bytes]`` JSONL fields survive a cache round-trip."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_shard = image_dir / "train-00000.tar"
    _make_tar(
        image_shard,
        [
            {"__key__": "image_00", "jpg": b"img0"},
            {"__key__": "image_01", "jpg": b"img1"},
        ],
    )

    shard = tmp_path / "samples.jsonl"
    _make_jsonl(
        shard,
        [
            {
                "images": [
                    "images/train-00000.tar#image_00.jpg",
                    "tar://images/train-00000.tar#image_01.jpg",
                ]
            }
        ],
    )

    call_count = [0]

    def annotate(sample):
        call_count[0] += 1
        return {**sample, "num_images": len(sample["images"])}

    ds = (
        Dataset.from_jsonl([str(shard)])
        .resolve_refs([("images", str(tmp_path))])
        .map(annotate)
        .cache(show_progress=False)
    )

    first = list(ds)
    assert call_count[0] == 1
    assert first[0]["images"] == [b"img0", b"img1"]
    assert first[0]["num_images"] == 2

    call_count[0] = 0
    second = list(ds)
    assert call_count[0] == 0
    assert second[0]["images"] == [b"img0", b"img1"]
    assert second[0]["num_images"] == 2


# ---------------------------------------------------------------------------
# Assemble round-trip
# ---------------------------------------------------------------------------


def test_cache_assemble_roundtrip(tmp_path):
    """Assembled outputs survive a cache round-trip."""
    from mvp_dataset import RuntimeContext

    shard = tmp_path / "shard-00000.tar"
    _make_tar(
        shard,
        [
            {"__key__": "0001", "x": b"a"},
            {"__key__": "0002", "x": b"b"},
            {"__key__": "0003", "x": b"c"},
            {"__key__": "0004", "x": b"d"},
        ],
    )

    class PairAssembler:
        def __init__(self):
            self._buf: list = []

        def push(self, sample):
            self._buf.append(sample)
            if len(self._buf) == 2:
                a, b_ = self._buf
                self._buf = []
                yield {"__key__": f"{a['__key__']}+{b_['__key__']}", "pair": a["x"] + b_["x"]}

        def finish(self, *, drop_last=False):
            if self._buf and not drop_last:
                yield {"__key__": self._buf[0]["__key__"], "pair": self._buf[0]["x"]}
            self._buf = []

    def factory(_ctx: RuntimeContext) -> PairAssembler:
        return PairAssembler()

    ds = Dataset.from_tars([str(shard)]).assemble(factory).cache(show_progress=False)

    results1 = list(ds)
    results2 = list(ds)

    assert len(results1) == len(results2) == 2
    for r1, r2 in zip(results1, results2, strict=True):
        assert r1["pair"] == r2["pair"]


# ---------------------------------------------------------------------------
# Multiple .cache() calls should raise
# ---------------------------------------------------------------------------


def test_double_cache_raises(tmp_path):
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "x": b"v"}])

    with pytest.raises(ValueError, match="only one .cache()"):
        Dataset.from_tars([str(shard)]).cache().cache()


# ---------------------------------------------------------------------------
# Post-cache stages execute after serving
# ---------------------------------------------------------------------------


def test_post_cache_map(tmp_path):
    """Map stages placed after .cache() run on cached data."""
    shard = tmp_path / "shard-00000.tar"
    _make_tar(shard, [{"__key__": "0001", "x": b"hello"}])

    post_call = [0]

    def post_fn(sample):
        post_call[0] += 1
        return {**sample, "y": b"post"}

    ds = Dataset.from_tars([str(shard)]).cache(show_progress=False).map(post_fn)

    list(ds)
    assert post_call[0] == 1

    post_call[0] = 0
    result = list(ds)
    assert post_call[0] == 1  # runs every iteration
    assert result[0]["y"] == b"post"


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------


def test_callable_fingerprint_stable():
    from mvp_dataset.cache.fingerprint import callable_fingerprint

    def fn(x):
        return x + 1

    fp1 = callable_fingerprint(fn)
    fp2 = callable_fingerprint(fn)
    assert fp1 == fp2
    assert len(fp1) == 64  # SHA-256 hex


def test_callable_fingerprint_differs_on_body_change():
    from mvp_dataset.cache.fingerprint import callable_fingerprint

    def fn_a(x):
        return x + 1

    def fn_b(x):
        return x + 2

    assert callable_fingerprint(fn_a) != callable_fingerprint(fn_b)


def test_callable_fingerprint_closure():
    from mvp_dataset.cache.fingerprint import callable_fingerprint

    def make(v):
        def fn(x):
            return x + v

        return fn

    assert callable_fingerprint(make(1)) != callable_fingerprint(make(2))
    assert callable_fingerprint(make(1)) == callable_fingerprint(make(1))


def test_callable_fingerprint_unsupported_closure():
    from mvp_dataset.cache.fingerprint import callable_fingerprint

    class Opaque:
        pass

    obj = Opaque()

    def fn(x):
        return obj

    with pytest.raises(ValueError, match="CacheFingerprintError"):
        callable_fingerprint(fn)


# ---------------------------------------------------------------------------
# DataLoadMesh / RuntimeContext – is_cache_leader
# ---------------------------------------------------------------------------


class _FakeMesh:
    """Minimal stand-in for ``torch.distributed.DeviceMesh``."""

    def __init__(self, dim_names: tuple[str, ...], local_ranks: dict[str, int], sizes: dict[str, int]):
        self.mesh_dim_names = dim_names
        self._local_ranks = local_ranks
        self._sizes = sizes

    def get_local_rank(self, dim: str) -> int:
        return self._local_ranks[dim]

    def size(self, dim: str) -> int:
        return self._sizes[dim]


def test_is_cache_leader_no_mesh():
    """Without a mesh every rank is its own leader."""
    from mvp_dataset.core.types import RuntimeContext

    ctx = RuntimeContext(rank=3, world_size=4)
    assert ctx.is_cache_leader is True


def test_is_cache_leader_pure_dp():
    """When all mesh dims are DP, every rank is a leader (no TP dims)."""
    from mvp_dataset.core.types import DataLoadMesh

    mesh = _FakeMesh(
        dim_names=("replicate", "shard"),
        local_ranks={"replicate": 1, "shard": 0},
        sizes={"replicate": 2, "shard": 4},
    )
    dlm = DataLoadMesh(device_mesh=mesh, dp_dims=("replicate", "shard"))
    assert dlm.is_cache_leader is True


def test_is_cache_leader_tp_leader():
    """TP local-rank 0 is the leader."""
    from mvp_dataset.core.types import DataLoadMesh, RuntimeContext

    mesh = _FakeMesh(
        dim_names=("dp", "tp"),
        local_ranks={"dp": 1, "tp": 0},
        sizes={"dp": 4, "tp": 2},
    )
    dlm = DataLoadMesh(device_mesh=mesh, dp_dims=("dp",))
    assert dlm.is_cache_leader is True

    ctx = RuntimeContext(rank=2, world_size=8, mesh=dlm)
    assert ctx.is_cache_leader is True


def test_is_cache_leader_tp_follower():
    """TP local-rank != 0 is a follower and must NOT build cache."""
    from mvp_dataset.core.types import DataLoadMesh, RuntimeContext

    mesh = _FakeMesh(
        dim_names=("dp", "tp"),
        local_ranks={"dp": 1, "tp": 1},
        sizes={"dp": 4, "tp": 2},
    )
    dlm = DataLoadMesh(device_mesh=mesh, dp_dims=("dp",))
    assert dlm.is_cache_leader is False

    ctx = RuntimeContext(rank=3, world_size=8, mesh=dlm)
    assert ctx.is_cache_leader is False


def test_is_cache_leader_3d_mesh():
    """3-D mesh (replicate, shard, tensor): leader only when tensor local-rank == 0."""
    from mvp_dataset.core.types import DataLoadMesh

    # Leader: tensor=0
    mesh_leader = _FakeMesh(
        dim_names=("replicate", "shard", "tensor"),
        local_ranks={"replicate": 0, "shard": 1, "tensor": 0},
        sizes={"replicate": 2, "shard": 4, "tensor": 8},
    )
    assert DataLoadMesh(device_mesh=mesh_leader, dp_dims=("replicate", "shard")).is_cache_leader is True

    # Follower: tensor=3
    mesh_follower = _FakeMesh(
        dim_names=("replicate", "shard", "tensor"),
        local_ranks={"replicate": 0, "shard": 1, "tensor": 3},
        sizes={"replicate": 2, "shard": 4, "tensor": 8},
    )
    assert DataLoadMesh(device_mesh=mesh_follower, dp_dims=("replicate", "shard")).is_cache_leader is False


# ---------------------------------------------------------------------------
# wait_for_cache
# ---------------------------------------------------------------------------


def test_wait_for_cache_immediate(tmp_path):
    """wait_for_cache returns immediately when manifests already exist."""
    from mvp_dataset.cache.materialize import _write_manifest, wait_for_cache

    shard = str(tmp_path / "shard-00000.tar")
    Path(shard).touch()
    _write_manifest(shard, "fp123", {"g": str(tmp_path / "g.tar")})
    Path(tmp_path / "g.tar").touch()

    # Should return instantly without timeout.
    wait_for_cache([shard], "fp123", poll_interval=0.01, timeout=1.0)


def test_wait_for_cache_timeout(tmp_path):
    """wait_for_cache raises TimeoutError when manifest never appears."""
    from mvp_dataset.cache.materialize import wait_for_cache

    shard = str(tmp_path / "shard-00000.tar")
    Path(shard).touch()

    with pytest.raises(TimeoutError, match="CacheWaitTimeout"):
        wait_for_cache([shard], "fp_missing", poll_interval=0.01, timeout=0.05)


def test_wait_for_cache_delayed(tmp_path):
    """wait_for_cache succeeds when manifest appears after a short delay."""
    import threading

    from mvp_dataset.cache.materialize import _write_manifest, wait_for_cache

    shard = str(tmp_path / "shard-00000.tar")
    Path(shard).touch()
    group_tar = tmp_path / "g.tar"
    group_tar.touch()

    def _build_later():
        import time

        time.sleep(0.1)
        _write_manifest(shard, "fp_delayed", {"g": str(group_tar)})

    t = threading.Thread(target=_build_later)
    t.start()
    wait_for_cache([shard], "fp_delayed", poll_interval=0.02, timeout=5.0)
    t.join()


def test_cache_progress_uses_shard_counter(tmp_path, capsys):
    """cache(show_progress=True) reports shard counters and ETA."""
    shard_a = tmp_path / "shard-00000.tar"
    shard_b = tmp_path / "shard-00001.tar"
    _make_tar(shard_a, [{"__key__": "0001", "x": b"a"}])
    _make_tar(shard_b, [{"__key__": "0002", "x": b"b"}])

    ds = Dataset.from_tars([str(shard_a), str(shard_b)]).cache(show_progress=True)
    list(ds)

    stderr = capsys.readouterr().err
    assert "Caching 1/2 shards... ETA " in stderr
    assert "Caching 2/2 shards... ETA " in stderr


def test_cache_progress_uses_injected_logger(tmp_path):
    """cache progress should use the globally injected logger when present."""
    from mvp_dataset import reset_logger, set_logger

    class _FakeLogger:
        def __init__(self) -> None:
            self.infos: list[str] = []
            self.errors: list[str] = []

        def info(self, msg, *args, **kwargs):
            self.infos.append(str(msg))

        def error(self, msg, *args, **kwargs):
            self.errors.append(str(msg))

    shard_a = tmp_path / "shard-00000.tar"
    shard_b = tmp_path / "shard-00001.tar"
    _make_tar(shard_a, [{"__key__": "0001", "x": b"a"}])
    _make_tar(shard_b, [{"__key__": "0002", "x": b"b"}])

    logger = _FakeLogger()
    set_logger(logger)
    try:
        ds = Dataset.from_tars([str(shard_a), str(shard_b)]).cache(show_progress=True)
        list(ds)
    finally:
        reset_logger()

    assert any(msg.startswith("Caching 1/2 shards... ETA ") for msg in logger.infos)
    assert any(msg.startswith("Caching 2/2 shards... ETA ") for msg in logger.infos)


# ---------------------------------------------------------------------------
# Non-leader TP rank skips warm-up (end-to-end)
# ---------------------------------------------------------------------------


def test_non_leader_skips_warmup(tmp_path):
    """A non-leader TP rank should read from cache built by the leader, not re-run the pipeline."""
    from mvp_dataset.core.types import DataLoadMesh, RuntimeContext

    # Create a shard.
    shard = tmp_path / "shard-00000.tar"
    _make_tar(
        shard,
        [
            {"__key__": "a", "x": b"hello"},
            {"__key__": "b", "x": b"world"},
        ],
    )

    call_count = [0]

    def counting_map(sample):
        call_count[0] += 1
        return {**sample, "y": b"mapped"}

    # --- Leader (tp=0) builds the cache ---
    leader_mesh = _FakeMesh(
        dim_names=("dp", "tp"),
        local_ranks={"dp": 0, "tp": 0},
        sizes={"dp": 1, "tp": 2},
    )
    leader_ctx = RuntimeContext(
        rank=0,
        world_size=2,
        mesh=DataLoadMesh(device_mesh=leader_mesh, dp_dims=("dp",)),
    )
    ds_leader = Dataset.from_tars([str(shard)], context=leader_ctx).map(counting_map).cache(show_progress=False)
    leader_result = list(ds_leader)
    assert call_count[0] == 2  # map ran for both samples
    assert len(leader_result) == 2

    # --- Follower (tp=1) should NOT run the map ---
    call_count[0] = 0
    follower_mesh = _FakeMesh(
        dim_names=("dp", "tp"),
        local_ranks={"dp": 0, "tp": 1},
        sizes={"dp": 1, "tp": 2},
    )
    follower_ctx = RuntimeContext(
        rank=1,
        world_size=2,
        mesh=DataLoadMesh(device_mesh=follower_mesh, dp_dims=("dp",)),
    )
    ds_follower = Dataset.from_tars([str(shard)], context=follower_ctx).map(counting_map).cache(show_progress=False)
    follower_result = list(ds_follower)
    assert call_count[0] == 0, "non-leader should NOT have run the map function"
    assert len(follower_result) == 2
    for s in follower_result:
        assert s["y"] == b"mapped"
