# Agent Integration Contract

Drop-in agents live under `agents/<name>/`.

Required file:
- `adapter.py` with `create_agent(config: dict) -> AgentAdapter`

Required adapter methods:
- `reset() -> None`
- `act(observation, deterministic: bool = False) -> action`
- `update(batch) -> Mapping[str, float]`
- `save(path: Path) -> None`

Optional file:
- `manifest.json` with metadata (`name`, `version`, `framework`, `description`).

Validation command:
```bash
python -m crlbench validate-agent --agent <name> --agents-dir agents
```

Run command:
```bash
python -m crlbench run-experiment \
  --experiment experiment_1_forgetting \
  --track toy \
  --env-family dm_control \
  --env-option vision_sequential_default \
  --agent <name> \
  --agents-dir agents \
  --tier smoke
```

Matrix command (all Experiment 1 roots):
```bash
python -m crlbench run-experiment1-matrix \
  --agent <name> \
  --agents-dir agents \
  --tier smoke
```

Pass agent hyperparameters:
```bash
python -m crlbench run-experiment \
  ... \
  --set learning_rate=0.005 \
  --set clip_epsilon=0.15
```
