# mm-loader

`mm-loader` is a maintainable and extensible multimodal data loading core.

Current scope (step 1): local tar shards, deterministic distributed shard splitting,
streaming sample shuffle, and basic iterator pipeline ops (`map`, `shuffle`, `batch`, `unbatch`).
