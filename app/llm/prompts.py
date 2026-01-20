from langchain_core.prompts import PromptTemplate

MAP_PROMPT = PromptTemplate.from_template("""
Write a concise summary of the following text:

{text}

CONCISE SUMMARY:
""")

REDUCE_PROMPT = PromptTemplate.from_template("""
You are an M&A expert.

Write a concise summary with 2-3 sentences covering key points.
Add a title to the summary.

{text}
""")

STUFF_PROMPT = PromptTemplate.from_template("""
You are an M&A expert.

Write a concise summary with 2-3 sentences covering key points.
Add a title to the summary.

{text}
""")
