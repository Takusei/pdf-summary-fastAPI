from langchain_community.document_loaders import PyPDFLoader


def summarize_single_pdf(file_path: str, agent) -> str:
    """Summarizes the text content of a PDF file."""
    print(f"Summarizing file: {file_path}")

    if file_path.lower().endswith(".pdf") is False:
        return "Error: The provided file is not a PDF."

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])

    if text:
        # Invoke the agent with the user's message
        agent_response = agent.invoke({"messages": [("user", text)]})
        reply_text = agent_response["messages"][-1].content
        return reply_text
    return "Error: Could not extract text from the PDF."
