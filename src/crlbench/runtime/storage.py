from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunStorageLayout:
    root: Path
    experiment: str
    track: str
    env_option: str
    seed: int
    run_id: str

    @property
    def run_dir(self) -> Path:
        return (
            self.root
            / self.experiment
            / self.track
            / self.env_option
            / f"seed_{self.seed}"
            / self.run_id
        )


def resolve_run_storage_layout(  # noqa: PLR0913
    *,
    root: Path | str,
    experiment: str,
    track: str,
    env_option: str,
    seed: int,
    run_id: str,
    create: bool = False,
) -> RunStorageLayout:
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}.")
    fields = {
        "experiment": experiment,
        "track": track,
        "env_option": env_option,
        "run_id": run_id,
    }
    for name, value in fields.items():
        if not value.strip():
            raise ValueError(f"{name} must be non-empty.")
    layout = RunStorageLayout(
        root=Path(root),
        experiment=experiment,
        track=track,
        env_option=env_option,
        seed=seed,
        run_id=run_id,
    )
    if create:
        layout.run_dir.mkdir(parents=True, exist_ok=False)
    return layout
