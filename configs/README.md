# Configs

Configs are loaded in this order:
1. Base config (from file path passed to CLI),
2. Optional `extends` chain (resolved recursively),
3. Optional `--layer` overlays (applied in CLI order),
4. Optional `--set` runtime overrides (`KEY=VALUE`, dotted path supported).

Supported file formats:
- `.json` (always),
- `.yaml` / `.yml` (requires `PyYAML` extra).

Override examples:
```bash
python -m crlbench validate-config \
  --config configs/base.json \
  --set budget.train_steps=50000 \
  --set tags='["dev","debug"]'
```

Layering example:
```bash
python -m crlbench resolve-config \
  --config configs/base.json \
  --layer configs/exp1.json \
  --layer configs/toy_dm_control.json \
  --out /tmp/resolved.json
```

Example:
```json
{
  "extends": "base.json",
  "run_name": "exp1_dmcontrol_toy_full",
  "budget": {
    "train_steps": 200000,
    "eval_interval_steps": 5000,
    "eval_episodes": 20
  },
  "tags": ["full"]
}
```
