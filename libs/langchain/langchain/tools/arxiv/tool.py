"""Tool for the Arxiv API."""

from typing import Optional, Type

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.arxiv import ArxivAPIWrapper
from langchain.sync_utils import make_async

class ArxivInput(BaseModel): 
    query: str = Field(description="search query to look up")


class ArxivQueryRun(BaseTool):
    """Tool that adds the capability to search using the Arxiv API."""

    name: str = "arxiv"
    description: str = (
        "A wrapper around Arxiv.org "
        "Useful for when you need to answer questions about Physics, Mathematics, "
        "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
        "Electrical Engineering, and Economics "
        "from scientific articles on arxiv.org. "
        "Input should be a search query."
    )
    api_wrapper: ArxivAPIWrapper = Field(default_factory=ArxivAPIWrapper)
    args_schema: Type[BaseModel] = ArxivInput

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Arxiv tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Arxiv tool asynchronously."""
        # raise NotImplementedError("ArxivAPIWrapper does not support async")
        return await make_async(self.api_wrapper.run)(query)
