"""Action and observation models for terminal mode.

Field for field the same as OpenEnv's envs/tbench2_env (Terminal-Bench 2) models at the pinned revision, class
names included, so the JSON schema in the model's system message and the JSON observations it reads match that
contract exactly. The private arena tasks appear to be served through tbench2_env (see docs/STRATEGY_v3.md), so
training on the same shapes trains the same reply habits.
"""

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class Tbench2Action(Action):
    """Action for interacting with a TB2 task session."""

    action_type: str = Field(default="exec")
    command: str = Field(default="")
    session_id: str | None = Field(default=None)
    block: bool = Field(default=True)
    wait_seconds: float | None = Field(default=None)
    file_path: str = Field(default="")
    content: str = Field(default="")


class Tbench2Observation(Observation):
    """Observation returned from the TB2 environment."""

    instruction: str = Field(default="")
    output: str = Field(default="")
    success: bool = Field(default=True)
    error: str = Field(default="")
    task_id: str = Field(default="")
    task_path: str = Field(default="")
    session_id: str | None = Field(default=None)
    action_type: str = Field(default="")
    info: dict = Field(default_factory=dict)


class Tbench2State(State):
    """Server-side state for a TB2 task."""

    task_id: str = Field(default="")
    task_path: str = Field(default="")
    session_id: str = Field(default="")
    terminal_ready: bool = Field(default=False)
    last_action_type: str = Field(default="")
    last_command: str = Field(default="")
    last_output: str = Field(default="")
