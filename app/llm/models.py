import os

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI, ChatOpenAI

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or None
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION") or None
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or None
AZURE_TOKEN_PROVIDER = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)


def initialize_model():
    if AZURE_OPENAI_ENDPOINT:
        return AzureChatOpenAI(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_ad_token_provider=AZURE_TOKEN_PROVIDER,
            temperature=0,
            timeout=1000,
            max_retries=3,
        )
    return ChatOpenAI(model="gpt-4.1-nano", temperature=0, timeout=10, max_tokens=1000)


def initialize_agent():
    # Configure model
    model = initialize_model()

    # Create agent
    agent = create_agent(
        model=model,
        system_prompt="You are a M&A document analysis assistant. "
        "Give 2~3 sentence summaries based on the user's input documents.",
    )
    return agent
