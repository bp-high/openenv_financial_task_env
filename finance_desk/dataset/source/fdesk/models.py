"""Action and observation models. One generic action serves every task."""

from typing import Any

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


class DeskAction(Action):
    tool: str = Field(
        description="One of: list, read, python, submit.",
    )
    args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            'Tool arguments. read: {"path": "file.csv"} or {"paths": ["a.csv", "b.json"]}. '
            'python: {"code": "print(1+1)"}. submit: {"answer": {...the documented keys...}}. list: {}.'
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _fold_stray_keys(cls, data: Any) -> Any:
        # Forgiving parse: {"tool": "read", "path": "x"} means {"tool": "read", "args": {"path": "x"}}.
        if isinstance(data, dict):
            known = {"tool", "args", "metadata"}
            stray = {k: v for k, v in data.items() if k not in known}
            if stray:
                data = {k: v for k, v in data.items() if k in known}
                args = dict(data.get("args") or {})
                for k, v in stray.items():
                    args.setdefault(k, v)
                data["args"] = args
        return data


class DeskObservation(Observation):
    output: str = Field(default="", description="Task brief at reset, otherwise the result of the last tool call.")
    steps_left: int = Field(default=0, description="Tool calls left before the episode ends with reward 0.")
