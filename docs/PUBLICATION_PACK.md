# Publication Pack

`export-publication-pack` produces a shareable benchmark bundle with:
- run-level summaries,
- experiment-level aggregate summaries,
- CSV + LaTeX tables,
- canonical figure specs (and optional PNG images),
- copied manifests and resolved configs,
- method metadata.

Example:
```bash
python -m crlbench export-publication-pack \
  --run-dir artifacts/<run_id_1> \
  --run-dir artifacts/<run_id_2> \
  --out-dir /tmp/publication_pack \
  --method agent=dreamer \
  --method benchmark=v1
```

Optional image rendering:
```bash
python -m crlbench export-publication-pack \
  --run-dir artifacts/<run_id_1> \
  --out-dir /tmp/publication_pack \
  --render-images
```

Notes:
- `--render-images` requires `matplotlib`.
- Without `--render-images`, `.plot.json` specs are still generated for all canonical plots.
