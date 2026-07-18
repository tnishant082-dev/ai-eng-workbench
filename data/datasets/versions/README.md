# Dataset versioning convention

```
data/datasets/versions/<version_id>/
  manifest.json   # schema, row count, checksum note
  <files>.csv
```

Promote a version by copying into `data/datasets/` and updating `AIWB_DATASET_PATH`,
or pass the versioned path to the train API. This is a **folder convention**, not DVC/LakeFS.
