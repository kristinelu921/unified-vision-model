# LVM conda environments

Exported from Kristine's miniconda install:

- `lvm` — Python 3.12 base env (minimal)
- `lvm310` — Python 3.10 working env used for LVM/JAX/TPU workflows (~13G installed)

## Recreate

```bash
# Full lock-style export (may need matching channels/platforms):
conda env create -f envs/environment-lvm310.yml

# Or history-only skeleton, then install pip deps:
conda env create -f envs/environment-lvm310-from-history.yml
conda run -n lvm310 pip install -r envs/requirements-lvm310-pip.txt
```

Also see `jflux/requirements.txt` for the curated JFLUX2 direct dependencies.
