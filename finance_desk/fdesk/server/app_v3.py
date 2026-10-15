from openenv.core.env_server.http_server import create_app

from fdesk.terminal.environment import TerminalDeskEnvironment
from fdesk.terminal.models import Tbench2Action, Tbench2Observation

app = create_app(
    TerminalDeskEnvironment,
    Tbench2Action,
    Tbench2Observation,
    env_name="finance_desk_terminal",
    max_concurrent_envs=8,
)
