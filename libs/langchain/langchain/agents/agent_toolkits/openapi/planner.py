"""Agent that interacts with OpenAPI APIs via a hierarchical planning approach."""
import json
import re
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import yaml
from pydantic import Field
from pydantic import BaseModel
from langchain.callbacks.manager import Callbacks

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.openapi.planner_prompt import (
    API_CONTROLLER_PROMPT,
    API_CONTROLLER_TOOL_DESCRIPTION,
    API_CONTROLLER_TOOL_NAME,
    API_ORCHESTRATOR_PROMPT,
    API_PLANNER_PROMPT,
    API_PLANNER_TOOL_DESCRIPTION,
    API_PLANNER_TOOL_NAME,
    PARSING_DELETE_PROMPT,
    PARSING_GET_PROMPT,
    PARSING_PATCH_PROMPT,
    PARSING_POST_PROMPT,
    REQUESTS_DELETE_TOOL_DESCRIPTION,
    REQUESTS_GET_TOOL_DESCRIPTION,
    REQUESTS_PATCH_TOOL_DESCRIPTION,
    REQUESTS_POST_TOOL_DESCRIPTION,
)
from langchain.agents.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import PromptTemplate
from langchain.requests import RequestsWrapper
from langchain.schema import BasePromptTemplate
from langchain.tools.base import BaseTool
from langchain.tools.requests.tool import BaseRequestsTool
from langchain.sync_utils import make_async
from langchain.json_utils import maybe_fix_json
from langchain.media_utils import send_medias_message, is_media
#
# Requests tools with LLM-instructed extraction of truncated responses.
#
# Of course, truncating so bluntly may lose a lot of valuable
# information in the response.
# However, the goal for now is to have only a single inference step.
MAX_RESPONSE_LENGTH = 5000


def _get_default_llm_chain(prompt: BasePromptTemplate) -> LLMChain:
    return LLMChain(
        llm=OpenAI(),
        prompt=prompt,
    )


def _get_default_llm_chain_factory(
    prompt: BasePromptTemplate,
) -> Callable[[], LLMChain]:
    """Returns a default LLMChain factory."""
    return partial(_get_default_llm_chain, prompt)


def _build_output_instructions(input: str):
    try:
        if isinstance(input, dict):
            json_obj = input
            res = ", ".join(json_obj.keys())
        elif isinstance(input, str):
            json_obj = json.loads(input)
            res = ", ".join(json_obj.keys())
        elif isinstance(input, (list, tuple)):
            res = ", ".join(input)
    except json.JSONDecodeError as e:
        res = input
    return res


class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests.get"
    description = REQUESTS_GET_TOOL_DESCRIPTION
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_GET_PROMPT)
    )

    def _run(self, text: str) -> str:
        data = maybe_fix_json(text)
        data_params = data.get("params")
        response = self.requests_wrapper.get(data["url"], params=data_params)
        response = response[: self.response_length]
        # return self.llm_chain.predict(
        #     response=response, instructions=data["output_instructions"]
        # ).strip()
        return response

    async def _arun(self, text: str) -> str:
        # raise NotImplementedError()
        # return await make_async(self._run)(text)
        data = maybe_fix_json(text)
        if is_media(data['url']):
            return f"Do not need to get the content of {data['url']}, itself is the result."
        data_params = data.get("params")
        response = await self.requests_wrapper.aget(data["url"], params=data_params)
        response = response[: self.response_length]

        response = str(await self.llm_chain.apredict(
            response=response, instructions=_build_output_instructions(data["output_instructions"]), stop=['<END_OF_PARSE>']
        )).strip()
        await send_medias_message(response)

        return response


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests.post"
    description = REQUESTS_POST_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_POST_PROMPT)
    )

    def _run(self, text: str) -> str:
        data = maybe_fix_json(text)
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]
        # return self.llm_chain.predict(
        #     response=response, instructions=data["output_instructions"]
        # ).strip()
        return response

    async def _arun(self, text: str) -> str:
        # raise NotImplementedError()
        # return await make_async(self._run)(text)
        data = maybe_fix_json(text)
        if is_media(data['url']):
            return f"Do not need to get the content of {data['url']}, itself is the result."
        response = await self.requests_wrapper.apost(data["url"], data["data"])
        response = response[: self.response_length]

        response = str(await self.llm_chain.apredict(
            response=response, instructions=_build_output_instructions(data["output_instructions"]), stop=['<END_OF_PARSE>']
        )).strip()
        await send_medias_message(response)

        return response


class RequestsPatchToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests.patch"
    description = REQUESTS_PATCH_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_PATCH_PROMPT)
    )

    def _run(self, text: str) -> str:
        data = maybe_fix_json(text)
        response = self.requests_wrapper.patch(data["url"], data["data"])
        response = response[: self.response_length]
        # return self.llm_chain.predict(
        #     response=response, instructions=data["output_instructions"]
        # ).strip()
        return response

    async def _arun(self, text: str) -> str:
        # raise NotImplementedError()
        # return await make_async(self._run)(text)
        data = maybe_fix_json(text)
        if is_media(data['url']):
            return f"Do not need to get the content of {data['url']}, itself is the result."
        response = await self.requests_wrapper.apatch(data["url"], data["data"])
        response = response[: self.response_length]

        response = str(await self.llm_chain.apredict(
            response=response, instructions=_build_output_instructions(data["output_instructions"]), stop=['<END_OF_PARSE>']
        )).strip()
        await send_medias_message(response)

        return response


class RequestsDeleteToolWithParsing(BaseRequestsTool, BaseTool):
    name = "requests.delete"
    description = REQUESTS_DELETE_TOOL_DESCRIPTION

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain: LLMChain = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_DELETE_PROMPT)
    )

    def _run(self, text: str) -> str:
        data = maybe_fix_json(text)
        response = self.requests_wrapper.delete(data["url"])
        response = response[: self.response_length]
        # return self.llm_chain.predict(
        #     response=response, instructions=data["output_instructions"]
        # ).strip()
        return response

    async def _arun(self, text: str) -> str:
        # raise NotImplementedError()
        # return await make_async(self._run)(text)
        data = maybe_fix_json(text)
        if is_media(data['url']):
            return f"Do not need to get the content of {data['url']}, itself is the result."
        response = await self.requests_wrapper.adelete(data["url"])
        response = response[: self.response_length]

        response = str(await self.llm_chain.apredict(
            response=response, instructions=_build_output_instructions(data["output_instructions"]), stop=['<END_OF_PARSE>']
        )).strip()
        await send_medias_message(response)

        return response

class ApiPlanner(BaseModel):
    llm_chain: LLMChain
    stop: Optional[List] = None

    def plan(self, *args: Any, callbacks: Callbacks = None, **kwargs: Any) -> str:
        """Given input, decided what to do."""
        kwargs['query'] = args[0]
        llm_response = self.llm_chain.run(stop=self.stop, callbacks=callbacks, **kwargs)
        return llm_response

    async def aplan(
        self, *args: Any, callbacks: Callbacks = None, **kwargs: Any
    ) -> str:
        """Given input, decided what to do."""
        kwargs['query'] = args[0]

        llm_response = await self.llm_chain.arun(
            stop=self.stop, callbacks=callbacks, **kwargs
        )
        return llm_response

#
# Orchestrator, planner, controller.
#
def _create_api_planner_tool(
    api_spec: ReducedOpenAPISpec, 
    llm: BaseLanguageModel, 
) -> Tool:
    endpoint_descriptions = [
        f"{name} {description}\n" for name, description, _ in api_spec.endpoints
    ]
    prompt = PromptTemplate(
        template=API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables={"endpoints": "- " + "- ".join(endpoint_descriptions)},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    planner = ApiPlanner(llm_chain=chain, stop=['<END_OF_PLAN>'])
    tool = Tool(
        name=API_PLANNER_TOOL_NAME,
        description=API_PLANNER_TOOL_DESCRIPTION,
        func=planner.plan,
        coroutine=planner.aplan
    )
    return tool


def _create_api_controller_agent(
    api_url: str,
    api_docs: str,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
) -> AgentExecutor:
    get_llm_chain = LLMChain(llm=llm, prompt=PARSING_GET_PROMPT)
    post_llm_chain = LLMChain(llm=llm, prompt=PARSING_POST_PROMPT)
    tools: List[BaseTool] = [
        RequestsGetToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=get_llm_chain
        ),
        RequestsPostToolWithParsing(
            requests_wrapper=requests_wrapper, llm_chain=post_llm_chain
        ),
    ]
    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "api_url": api_url,
            "api_docs": api_docs,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def _create_api_controller_tool(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
) -> Tool:
    """Expose controller as a tool.

    The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to
    constrain the context.
    """

    base_url = api_spec.servers[0]["url"]  # TODO: do better.

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST|PATCH|DELETE)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            docs = endpoint_docs_by_name.get(endpoint_name)
            if not docs:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")
            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"

        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm)
        return agent.run(plan_str)
    
    async def _acreate_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST|PATCH|DELETE)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            docs = endpoint_docs_by_name.get(endpoint_name)
            if not docs:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")
            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"

        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm)
        return await agent.arun(plan_str)

    return Tool(
        name=API_CONTROLLER_TOOL_NAME,
        func=_create_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION,
        coroutine=_acreate_and_run_api_controller_agent
    )


def create_openapi_agent(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    shared_memory: Optional[ReadOnlySharedMemory] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = True,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Instantiate API planner and controller for a given spec.

    Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller,
    rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.
    """
    tools = [
        _create_api_planner_tool(api_spec, llm),
        _create_api_controller_tool(api_spec, requests_wrapper, llm),
    ]
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt, memory=shared_memory),
        allowed_tools=[tool.name for tool in tools],
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
