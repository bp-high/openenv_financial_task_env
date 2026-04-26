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
        # The env-server wraps responses as {observation: {...}, reward, done}.
        # Older openenv-core versions returned the obs at the top level, so we
        # fall back to using the whole payload if no 'observation' key is present.
        obs_data = payload.get("observation", payload) if isinstance(payload, dict) else {}
        obs = FinancialObservation(**obs_data)
        reward = payload.get("reward", obs.reward) if isinstance(payload, dict) else obs.reward
        done = payload.get("done", obs.done) if isinstance(payload, dict) else obs.done
        return StepResult(
            observation=obs,
            reward=reward if isinstance(reward, (int, float)) else 0.0,
            done=bool(done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> Any:
        from openenv.core.env_server.types import State

        return State(**payload)
