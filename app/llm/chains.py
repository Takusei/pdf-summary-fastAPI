from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.llm.prompts import MAP_PROMPT, REDUCE_PROMPT, STUFF_PROMPT


def build_map_chain(llm: ChatOpenAI):
    return MAP_PROMPT | llm | StrOutputParser()


def build_reduce_chain(llm: ChatOpenAI):
    return REDUCE_PROMPT | llm | StrOutputParser()


def build_stuff_chain(llm: ChatOpenAI):
    return STUFF_PROMPT | llm | StrOutputParser()
