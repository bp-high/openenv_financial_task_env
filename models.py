"""Typed Pydantic models for the Financial Task Environment."""

from typing import Any, Dict

from pydantic import Field

from openenv.core.env_server.types import Action, Observation, State


class FinancialAction(Action):
    """Action model for the Financial Task Environment.

    Agents interact by executing Python code to read/modify xlsx files,
    or by submitting a text answer / file path.
    """

    action_type: str = Field(
        description="Action type: 'code' to execute Python, 'submit' for text answer, 'submit_file' for xlsx"
    )
    content: str = Field(
        description="Python code when action_type='code', text answer for 'submit', file path for 'submit_file'"
    )


class FinancialObservation(Observation):
    """Observation model for the Financial Task Environment.

    Contains the task description, financial data, and feedback from
    the environment after each action.
    """

    task_id: str = Field(default="", description="Current task identifier")
    task_description: str = Field(default="", description="Task instructions")
    financial_data: str = Field(default="", description="Financial data / xlsx summary")
    difficulty: str = Field(default="", description="Task difficulty: easy, medium, or hard")
    feedback: str = Field(default="", description="Feedback on the last action taken")
    current_step: int = Field(default=0, description="Current step number in the episode")
    max_steps: int = Field(default=15, description="Maximum steps allowed per episode")
    task_type: str = Field(default="", description="Type of financial task: QA or MODIFY")
    source_file: str = Field(default="", description="Path to the working xlsx file")
    available_tasks: str = Field(
        default="",
        description="Comma-separated list of available task IDs (shown on reset)",
    )
