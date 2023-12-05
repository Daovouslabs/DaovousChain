"""Tool for interacting with a single API with natural language definition."""


from typing import Any, Optional
import json
from langchain_core.language_models import BaseLanguageModel

from langchain.agents.tools import Tool
from langchain_core.tools import StructuredTool
from langchain.chains.api.openapi.chain import OpenAPIEndpointChain
from langchain.tools.openapi.utils.api_models import APIOperation
from langchain.tools.openapi.utils.openapi_utils import OpenAPISpec
from langchain.utilities.requests import Requests
from pydantic.v1 import create_model, BaseModel

class NLATool(Tool):
    """Natural Language API Tool."""

    @property
    def args(self) -> dict:
        """The tool's input arguments."""
        if self.args_schema is not None:
            return self.args_schema.schema()["properties"]
        # For backwards compatibility, if the function signature is ambiguous,
        # assume it takes a single string input.
        return {"instructions": {"type": "string", "title": "instructions"}}

    @classmethod
    def from_open_api_endpoint_chain(
        cls, chain: OpenAPIEndpointChain, api_title: str
    ) -> "NLATool":
        """Convert an endpoint chain to an API endpoint tool."""
        expanded_name = (
            f'{api_title.replace(" ", "_")}.{chain.api_operation.operation_id}'
        )
        tool_desc = chain.api_operation.description.replace('\n', ' ')
        description = (
            f"I'm an AI from {api_title}. Instruct what you want,"
            " and I'll assist via an API with description:"
            f" {tool_desc}"
        )
        return cls(name=expanded_name, func=chain.run, coroutine=chain.arun, description=description)

    @classmethod
    def from_llm_and_method(
        cls,
        llm: BaseLanguageModel,
        path: str,
        method: str,
        spec: OpenAPISpec,
        requests: Optional[Requests] = None,
        verbose: bool = False,
        return_intermediate_steps: bool = False,
        **kwargs: Any,
    ) -> "NLATool":
        """Instantiate the tool from the specified path and method."""
        api_operation = APIOperation.from_openapi_spec(spec, path, method)

        # _args = {p.name: p for p in api_operation.properties}
        # _args = {}
        # for p in api_operation.properties:
        #     _p: dict = json.loads(p.json())
        #     _p_name = _p.pop('name')
        #     _p.pop('location')
        #     _args[_p_name] = (type(p.type), p.default)
        # _args_schema = create_model("args_schema", **_args)

        chain = OpenAPIEndpointChain.from_api_operation(
            api_operation,
            llm,
            requests=requests,
            verbose=verbose,
            return_intermediate_steps=return_intermediate_steps,
            **kwargs,
        )
        return cls.from_open_api_endpoint_chain(chain, spec.info.title)
