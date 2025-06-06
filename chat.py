import asyncio
import logging
import os


from dotenv import load_dotenv


from semantic_kernel.connectors.ai.open_ai import (
    AzureAISearchDataSource,
    AzureChatCompletion,
    AzureChatPromptExecutionSettings,
    ExtraBody,
)
from semantic_kernel.connectors.memory.azure_cognitive_search.azure_ai_search_settings import AzureAISearchSettings

from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.utils.logging import setup_logging
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)

from plugins.automation_plugin import AutomationPlugin
from plugins.lights_plugin import LightsPlugin


# Load the .env file
load_dotenv()


async def main():

    setup_logging()
    logging.getLogger("kernel").setLevel(logging.DEBUG)

    settings = AzureChatPromptExecutionSettings()
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

    chat_completion = AzureChatCompletion(
        deployment_name = os.getenv("DEPLOYMENT_NAME"),
        api_key=os.getenv("API_KEY"),
        endpoint=os.getenv("ENDPOINT"),
        api_version="2025-04-01-preview"
    )

    agent = ChatCompletionAgent(
        service=chat_completion,
        name="Assistant",
        instructions="You are a helpful assistant.",
        plugins=[LightsPlugin(), AutomationPlugin()],
        arguments=KernelArguments(settings)
    )

    # Create a history of the conversation
    history = ChatHistory()
    history.add_system_message("I am an AI assistant here to answer your questions.")

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