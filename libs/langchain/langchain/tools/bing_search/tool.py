"""Tool for the Bing search API."""

from typing import Optional

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.bing_search import BingSearchAPIWrapper
from langchain.sync_utils import make_async

class BingSearchRun(BaseTool):
    """Tool that adds the capability to query the Bing search API."""

    name: str = "bing_search"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: BingSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # raise NotImplementedError("BingSearchRun does not support async")
        return await make_async(self.api_wrapper.run)(query)


class BingSearchResults(BaseTool):
    """Tool that has capability to query the Bing Search API and get back json."""

    name: str = "Bing Search Results JSON"
    description: str = (
        "A wrapper around Bing Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query. Output is a JSON array of the query results"
    )
    num_results: int = 4
    api_wrapper: BingSearchAPIWrapper

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(self.api_wrapper.results(query, self.num_results))

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # raise NotImplementedError("BingSearchResults does not support async")
        return str(await make_async(self.api_wrapper.results)(query, self.num_results))
