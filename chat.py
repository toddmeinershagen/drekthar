import asyncio
import json
import logging
import os
from pydantic import SecretStr, HttpUrl
from dotenv import load_dotenv


from semantic_kernel.connectors.ai.open_ai import (
    AzureAISearchDataSource,
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
    AzureTextEmbedding,
    ExtraBody,
)

from semantic_kernel.memory import SemanticTextMemory
from semantic_kernel.core_plugins import TextMemoryPlugin
from semantic_kernel.connectors.memory.azure_cognitive_search.azure_cognitive_search_memory_store import AzureCognitiveSearchMemoryStore
from semantic_kernel.connectors.memory.azure_ai_search import AzureAISearchCollection
from semantic_kernel.connectors.memory.azure_cognitive_search.azure_ai_search_settings import AzureAISearchSettings

from semantic_kernel.kernel import Kernel, KernelPlugin
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions import FunctionResult, KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from plugins.automation_plugin import AutomationPlugin
from plugins.lights_plugin import LightsPlugin

from semantic_kernel.filters import FilterTypes, AutoFunctionInvocationContext


# Load the .env file
load_dotenv()

logger = logging.getLogger(__name__)
    

async def main():

    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    search_settings = AzureAISearchSettings(
        api_key = os.getenv("SEARCH_ADMIN_KEY"),
        endpoint = os.getenv("SEARCH_ENDPOINT"),
        index_name = "drekthar"
    )
    az_source = AzureAISearchDataSource.from_azure_ai_search_settings(azure_ai_search_settings=search_settings)
    extra = ExtraBody(data_sources=[az_source])
    settings = AzureChatPromptExecutionSettings(extra_body=extra)

    chat_completion = AzureChatCompletion(
        deployment_name = os.getenv("DEPLOYMENT_NAME"),
        api_key=SecretStr(os.getenv("LLM_API_KEY")),
        endpoint=HttpUrl(os.getenv("LLM_ENDPOINT")),
        api_version="2025-04-01-preview"
    )
    
    kernel = Kernel()

    # A filter is a piece of custom code that runs at certain points in the process
    # this sample has a filter that is called during Auto Function Invocation
    # this filter will be called for each function call in the response.
    # You can name the function itself with arbitrary names, but the signature needs to be:
    # `context, next`
    # You are then free to run code before the call to the next filter or the function itself.
    # if you want to terminate the function calling sequence. set context.terminate to True
    @kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
    async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
        """A filter that will be called for each function call in the response."""
        print("\nAuto function invocation filter")
        print(f"Function: {context.function.name}")
        print(f"Request sequence: {context.request_sequence_index}")
        print(f"Function sequence: {context.function_sequence_index}")

        # as an example
        function_calls = context.chat_history.messages[-1].items
        print(f"Number of function calls: {len(function_calls)}")
        # if we don't call next, it will skip this function, and go to the next one
        await next(context)


    # Create the chat completion agent
    agent = ChatCompletionAgent(
        kernel=kernel,
        service=chat_completion,
        name="Assistant",
        instructions='''
        You are an AI assistant that helps people find information.
        ''',
        plugins=[ LightsPlugin(), AutomationPlugin() ],
        arguments=KernelArguments(settings=settings)
    )
    
    prompt = """        
    # Context
    {{search.recall ask=$input collection="drekthar" relevance="0.85" limit="3"}}

    {{$input}}
    """

    # Create a history of the conversation
    history = ChatHistory()


    # Initiate a back-and-forth chat
    userInput = None
    while True:
        userInput = input("User > ")

        if userInput == "exit":
            break

        history.add_user_message(userInput)

        response = await agent.get_response(messages=history.messages)

        print(f"Assistant > {response.content}")

        history.add_message(response.message)


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())