from langchain_core.prompts import PromptTemplate

MAP_PROMPT = PromptTemplate.from_template("""
Write a concise summary of the following text:

{text}

CONCISE SUMMARY:
""")

REDUCE_PROMPT = PromptTemplate.from_template("""
You are an M&A expert.

Write a concise summary with 2-3 sentences covering key points from the following text:

*ATTENTION*: All below is the content to summarize.
{text}
""")

STUFF_PROMPT = PromptTemplate.from_template("""
You are an M&A expert.

Write a concise summary with 2-3 sentences covering key points from the following text:

*ATTENTION*: All below is the content to summarize.
{text}
""")
