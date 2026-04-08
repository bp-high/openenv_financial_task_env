"""Financial Task Environment client."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult, StateT

from models import FinancialAction, FinancialObservation


class FinancialTaskEnv(EnvClient["FinancialAction", "FinancialObservation", StateT]):
    """Client for connecting to a Financial Task Environment server.

    Example (async)::

        async with FinancialTaskEnv(base_url="http://localhost:8000") as env:
            result = await env.reset(task_id="task_1")
            print(result.observation.task_description)
            result = await env.step(FinancialAction(action_type="submit", content="42"))
            print(result.reward)

    Example (sync)::

        with FinancialTaskEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset(task_id="task_1")
            result = env.step(FinancialAction(action_type="submit", content="42"))
    """

    def _step_payload(self, action: FinancialAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FinancialObservation]:
        obs = FinancialObservation(**payload)
        return StepResult(
            observation=obs,
            reward=obs.reward if isinstance(obs.reward, (int, float)) else 0.0,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Any:
        from openenv.core.env_server.types import State

        return State(**payload)
