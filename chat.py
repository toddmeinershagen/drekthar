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
from semantic_kernel.contents.chat_history import ChatHistory, ChatMessageContent
from semantic_kernel.functions import FunctionResult, KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from plugins.automation_plugin import AutomationPlugin
from plugins.lights_plugin import LightsPlugin

from semantic_kernel.filters import FilterTypes, AutoFunctionInvocationContext


# Load the .env file
load_dotenv()

import http.client as http_client
http_client.HTTPConnection.debuglevel = 1

async def main():

    setup_logging()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    search_settings = AzureAISearchSettings(
        api_key = os.getenv("SEARCH_ADMIN_KEY"),
        endpoint = os.getenv("SEARCH_ENDPOINT"),
        index_name = "drekthar"
    )
    az_source = AzureAISearchDataSource.from_azure_ai_search_settings(azure_ai_search_settings=search_settings)
    extra = ExtraBody(data_sources=[az_source])
    # extra = { "data_sources": [az_source]}
    settings = AzureChatPromptExecutionSettings(
        service_id="chat", 
        max_tokens=800,
        temperature=0.0,
        top_p=0.95,
        extra_body=extra
    )
    # settings.extra_body = extra
    # settings = AzureChatPromptExecutionSettings()
    # settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
    settings.function_choice_behavior = None

    chat_completion = AzureChatCompletion(
        service_id="chat",
        deployment_name = os.getenv("DEPLOYMENT_NAME"),
        api_key=SecretStr(os.getenv("LLM_API_KEY")),
        endpoint=HttpUrl(os.getenv("LLM_ENDPOINT")),
        api_version="2024-05-01-preview"
    )
    
    kernel = Kernel()
    kernel.add_service(chat_completion)

    chat_function = kernel.add_function(
        prompt="{{$chat_history}}{{$user_input}}",
        plugin_name="ChatBot",
        function_name="Chat",
    )

    # A filter is a piece of custom code that runs at certain points in the process
    # this sample has a filter that is called during Auto Function Invocation
    # this filter will be called for each function call in the response.
    # You can name the function itself with arbitrary names, but the signature needs to be:
    # `context, next`
    # You are then free to run code before the call to the next filter or the function itself.
    # if you want to terminate the function calling sequence. set context.terminate to True
    # @kernel.filter(FilterTypes.AUTO_FUNCTION_INVOCATION)
    # async def auto_function_invocation_filter(context: AutoFunctionInvocationContext, next):
    #     """A filter that will be called for each function call in the response."""
    #     print("\nAuto function invocation filter")
    #     print(f"Function: {context.function.name}")
    #     print(f"Request sequence: {context.request_sequence_index}")
    #     print(f"Function sequence: {context.function_sequence_index}")

    #     # as an example
    #     function_calls = context.chat_history.messages[-1].items
    #     print(f"Number of function calls: {len(function_calls)}")
    #     # if we don't call next, it will skip this function, and go to the next one
    #     await next(context)


    # Create the chat completion agent
    # agent = ChatCompletionAgent(
    #     # kernel=kernel,
    #     service=chat_completion,
    #     name="Assistant",
    #     instructions='''
    #     You are an AI assistant that helps people find information.
    #     ''',
    #     # plugins=[ LightsPlugin(), AutomationPlugin() ],
    #     arguments=KernelArguments(settings=settings)
    # )

    # Create a history of the conversation
    history = ChatHistory()


    # Initiate a back-and-forth chat
    userInput = None
    while True:
        userInput = input("User > ")

        if userInput == "exit":
            break

        history.add_user_message(userInput)

        # clean_messages = [
        #     ChatMessageContent( 
        #         role = m.role, 
        #         content = m.content, 
        #         metadata = m.metadata
        #     )
        #     for m in history.messages
        # ]
        # response = await agent.get_response(messages=clean_messages)
        # response = await agent.get_response(messages=history.messages)

        arguments = KernelArguments(settings=settings)
        arguments["user_input"] = userInput
        arguments["chat_history"] = history

        response = await kernel.invoke(
            chat_function, 
            arguments=arguments
        )

        print(f"Assistant > {response}")

        history.add_assistant_message(str(response))


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())