# Drop-In Agents

Each agent lives in `agents/<agent_name>/` and must expose:

- `adapter.py` with:
  - `create_agent(config: dict) -> AgentAdapter`

Optional:
- `manifest.json` with metadata (`name`, `version`, `framework`, etc.).

Minimal example:

```python
def create_agent(config: dict):
    return MyAgent(config)
```

Validate locally:

```bash
python -m crlbench validate-agent --agent random
```

Built-in baselines:
- `random`: random discrete actions.
- `ppo_baseline`: lightweight discrete PPO-style baseline.
- `ppo_continuous_baseline`: PyTorch PPO with MLP actor/critic for continuous control.

For `ppo_continuous_baseline`, install torch:

```bash
python -m pip install -e ".[torch]"
```
