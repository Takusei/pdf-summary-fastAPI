from langchain_core.prompts import PromptTemplate

MAP_PROMPT = PromptTemplate.from_template("""
You are an expert summarizer.

Task:
- Summarize the following text chunk
- Focus only on key facts and ideas explicitly stated
- Do NOT add interpretation or conclusions beyond the text
- Write 1-2 concise sentences

*ATTENTION*: The text below is a PARTIAL chunk of a larger document.
{text}

CHUNK SUMMARY:
""")

REDUCE_PROMPT = PromptTemplate.from_template("""
You are an expert summarizer.

The text below consists of PARTIAL SUMMARIES from different sections of the same document.

Task:
- Merge these summaries into ONE coherent final summary
- Eliminate redundancy and repeated points
- Identify the most important GLOBAL themes
- Write 2-3 concise sentences for the summary
- Add a short, informative title on the first line

Format:
Title: <title>
Summary: <2-3 sentences>

*ATTENTION*: The text below is NOT raw document content.
{text}
""")

STUFF_PROMPT = PromptTemplate.from_template("""
You are an M&A professional experienced in reviewing transaction-related documents.

Context (metadata, NOT part of the content):
- File name: {file_name}

Task:
- Write a concise executive-level summary covering the key points relevant to an M&A engagement
- Focus on aspects such as purpose, scope, structure, risks, or implications if present
- Use 2-3 sentences
- Do NOT introduce information not explicitly supported by the text

Format:
Summary: <2-3 sentences>
*ATTENTION*: The text below is the FULL content to summarize.
{text}
""")
