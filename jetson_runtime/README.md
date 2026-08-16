# `jetson_runtime`

This package is the **deployment contract** and (from Task S.4) the runtime.

`policy_contract.json` is a **generated artifact** — written by
`scripts/generate_policy_contract.py`, which re-derives every field from primary
sources. **Never hand-edit it.** Run the generator instead, and
`--check` to verify it has not drifted.

`contract.py` is a thin typed loader over that JSON. It contains **no policy
number literals**, by design and by test.

**Import note:** `setup.cfg` packages only `mini_bdx`, so this package is
importable only with the repo root as cwd or on `PYTHONPATH`:

```bash
cd /path/to/Open_Duck_Mini_Jetson
python3 -c "from jetson_runtime import contract as C; print(C.OBS_DIM, C.ACTION_DIM)"
```

Field-by-field sourcing: [`docs/jetson-mod/sim2real/deployment_contract.md`](../docs/jetson-mod/sim2real/deployment_contract.md).
