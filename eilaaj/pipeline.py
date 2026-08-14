import os

from langchain_core.prompts import ChatPromptTemplate

from eilaaj import config
from eilaaj.document_loader import load_documents
from eilaaj.splitter import split_into_chunks
from eilaaj.llm import get_llm
from eilaaj.vector_store import (
    build_vector_store,
    load_vector_store,
    vector_store_exists,
    get_retriever,
)


def _friendly_source_name(source_path: str) -> str:
    """Turn a file path like 'data/boericke_materia_medica.pdf' into 'Boericke Materia Medica'."""
    if not source_path:
        return "Unknown source"
    base = os.path.splitext(os.path.basename(source_path))[0]
    return base.replace("_", " ").replace("-", " ").title()


def build_vector_store_for_data(data_path: str = config.DATA_PATH):
    """Load + split + embed everything in the data folder, reusing a saved index if present."""
    if vector_store_exists():
        print("Loading existing vector store...")
        return load_vector_store()

    print("No saved vector store found, building a new one...")
    documents = load_documents(data_path)
    if not documents:
        print(f"No documents found in '{data_path}'. Add files there and run again.")
        return None

    chunks = split_into_chunks(documents)
    print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunks.")

    vector_store = build_vector_store(chunks)
    print(f"Vector store built and saved to '{config.PERSIST_DIRECTORY}'.")
    return vector_store


def format_history(history: list, max_turns: int = 6) -> str:
    """Format the last few chat turns as plain text for the prompt."""
    if not history:
        return "(this is the first message in the conversation)"

    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        speaker = "User" if turn["sender"] == "user" else "E-Ilaaj"
        lines.append(f"{speaker}: {turn['message']}")
    return "\n".join(lines)


def query_rag(query_text: str, history: list = None) -> str:
    """Answer a user's message using retrieved context and recent chat history.

    Once the user has sent enough messages (config.REPORT_TRIGGER_TURNS),
    switch from asking clarifying questions to producing the final report.
    """
    llm = get_llm()
    history = history or []
    history_text = format_history(history)

    user_turns_so_far = sum(1 for turn in history if turn["sender"] == "user")
    is_report_turn = (user_turns_so_far + 1) >= config.REPORT_TRIGGER_TURNS
    template = config.REPORT_PROMPT_TEMPLATE if is_report_turn else config.SYSTEM_PROMPT_TEMPLATE

    vector_store = load_vector_store() if vector_store_exists() else None
    if not vector_store:
        # No knowledge base ingested yet — fall back to the raw LLM, still history-aware.
        instruction = (
            "Now produce the final structured report (Disease Overview, Indicated "
            "Remedy, Daily Routine, Diet, Disclaimer)."
            if is_report_turn
            else "Reply briefly (max 3-4 sentences, one question at a time)."
        )
        prompt = f"Conversation so far:\n{history_text}\n\n{instruction}\n\nLatest message: {query_text}"
        response = llm.invoke(prompt)
        return response.content

    retriever = get_retriever(vector_store)
    results = retriever.invoke(query_text)

    if results:
        blocks = []
        for doc in results:
            source_path = doc.metadata.get("source", "")
            source_name = _friendly_source_name(source_path)
            blocks.append(f"[Source: {source_name}]\n{doc.page_content}")
        context_text = "\n\n---\n\n".join(blocks)
    else:
        context_text = "No specific context found in the knowledge base."

    prompt = ChatPromptTemplate.from_template(template).format(
        context=context_text, question=query_text, history=history_text
    )
    response = llm.invoke(prompt)
    return response.content