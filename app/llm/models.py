from langchain.agents import create_agent
from langchain_openai import ChatOpenAI


def initialize_model():
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
