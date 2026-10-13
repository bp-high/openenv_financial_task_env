from openenv.core.env_server.http_server import create_app

from fdesk.environment import FinanceDeskEnvironment
from fdesk.models import DeskAction, DeskObservation

app = create_app(
    FinanceDeskEnvironment,
    DeskAction,
    DeskObservation,
    env_name="finance_desk",
    max_concurrent_envs=8,
)
