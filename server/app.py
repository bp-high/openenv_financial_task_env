"""FastAPI application for the Financial Task Environment."""

from openenv.core.env_server.http_server import create_app

from models import FinancialAction, FinancialObservation
from server.financial_environment import FinancialEnvironment

app = create_app(
    FinancialEnvironment,
    FinancialAction,
    FinancialObservation,
    env_name="financial_task_env",
)


def main() -> None:
    """Entry point for direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
