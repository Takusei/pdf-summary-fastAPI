import asyncio

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_summary_with_map_reduce(docs: list, llm: ChatOpenAI) -> str:
    """Generates a summary of the provided documents using a map-reduce strategy."""
    # 1. Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Split document into {len(all_splits)} sub-documents.")

    # 2. Create a map-reduce summarization chain using LCEL
    # Prompt for mapping step
    map_prompt = """
    Write a concise summary of the following text:
    "{text}"
    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate.from_template(map_prompt)
    map_chain = map_prompt_template | llm

    # Prompt for combining step
    combine_prompt = """
    You are an M&A expert. Write a concise summary with 2~3 sentences of the following text that covers the key points.
    Add a title to the summary.
    ```{text}```
    """
    combine_prompt_template = PromptTemplate.from_template(combine_prompt)
    reduce_chain = combine_prompt_template | llm

    # 3. Get the total summary
    map_summaries = map_chain.batch(all_splits)
    combined_summary_input = "\n".join([summary.content for summary in map_summaries])

    total_summary = reduce_chain.invoke({"text": combined_summary_input})

    return total_summary.content


def get_summary_with_stuff(docs: list, llm: ChatOpenAI) -> str:
    """Generates a summary of the provided documents using the 'stuff' method."""
    prompt_template = """
    You are an M&A expert. Write a concise summary with 2~3 sentences of the following text that covers the key points.
    Add a title to the summary.
    ```{text}```
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # Create a chain that "stuffs" all documents into the prompt
    chain = prompt | llm | StrOutputParser()

    # Combine all documents into a single string
    docs_content = "".join([doc.page_content for doc in docs])

    # Invoke the chain with the combined content
    summary = chain.invoke({"text": docs_content})
    return summary


def summarize_single_pdf(
    file_path: str,
    llm: ChatOpenAI = ChatOpenAI(
        model="gpt-4.1-nano", temperature=0, timeout=10, max_tokens=1000
    ),
    method: str = "stuff",
) -> str:
    """Summarizes the text content of a PDF file using a map-reduce strategy."""
    print(f"Summarizing file: {file_path}")

    if not file_path.lower().endswith(".pdf"):
        return "Error: The provided file is not a PDF."

    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        return f"Error loading PDF: {e}"

    if not docs:
        return "Error: The PDF file is empty or could not be read."

    if method == "auto":
        print("Using automatic method selection for summarization.")
        if len(docs) > 20:
            print("Document is large, using map-reduce for summarization.")
            summary = get_summary_with_map_reduce(docs, llm)
        else:
            print("Document is small, using 'stuff' method for summarization.")
            summary = get_summary_with_stuff(docs, llm)
    elif method == "map-reduce":
        print("Using map-reduce method for summarization.")
        summary = get_summary_with_map_reduce(docs, llm)
    else:
        print("Using 'stuff' method for summarization.")
        summary = get_summary_with_stuff(docs, llm)

    return summary


async def summarize_single_pdf_async(file_path: str, semaphore: asyncio.Semaphore):
    """
    Asynchronous wrapper for the summarize_single_pdf function, controlled by a semaphore.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        # Run the synchronous summarize_single_pdf function in a separate thread
        summary = await loop.run_in_executor(None, summarize_single_pdf, file_path)
        return summary
