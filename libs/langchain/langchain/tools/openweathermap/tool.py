"""Tool for the OpenWeatherMap API."""

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.pydantic_v1 import Field
from langchain.tools.base import BaseTool
from langchain.utilities import OpenWeatherMapAPIWrapper
from langchain.sync_utils import make_async

class OpenWeatherMapQueryRun(BaseTool):
    """Tool that adds the capability to query using the OpenWeatherMap API."""

    api_wrapper: OpenWeatherMapAPIWrapper = Field(
        default_factory=OpenWeatherMapAPIWrapper
    )

    name: str = "OpenWeatherMap"
    description: str = (
        "A wrapper around OpenWeatherMap API. "
        "Useful for fetching current weather information for a specified location. "
        "Input should be a location string (e.g. London,GB)."
    )

    def _run(
        self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the OpenWeatherMap tool."""
        return self.api_wrapper.run(location)

    async def _arun(
        self,
        location: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the OpenWeatherMap tool asynchronously."""
        # raise NotImplementedError("OpenWeatherMapQueryRun does not support async")
        return await make_async(self.api_wrapper.run)(location)
