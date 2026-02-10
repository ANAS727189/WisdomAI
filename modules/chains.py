"""Chain assembly — wires together the LLM, retriever, memory, and prompts.

Uses ``ConversationalRetrievalChain`` which internally runs TWO chains:

1. **Condense-question chain** – rewrites the latest user message into a
   standalone question using the chat history (prompt from ``prompts.py``).
2. **Stuff-docs QA chain** – retrieves relevant chunks, stuffs them into
   the QA prompt (also from ``prompts.py``), and generates the answer.

Memory is automatically updated after each call.
"""

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

from modules.prompts import get_qa_prompt, get_condense_question_prompt


def build_conversational_chain(
    llm,
    retriever,
    memory: ConversationBufferWindowMemory,
) -> ConversationalRetrievalChain:
    """Return a ready-to-use conversational RAG chain.

    Parameters
    ----------
    llm : ChatOpenAI (or compatible)
    retriever : VectorStoreRetriever
    memory : ConversationBufferWindowMemory (from ``modules.memory``)
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=get_condense_question_prompt(),
        combine_docs_chain_kwargs={"prompt": get_qa_prompt()},
        return_source_documents=True,
        verbose=False,
    )


def ask_question(chain: ConversationalRetrievalChain, question: str) -> dict:
    """Run *question* through the chain and return a tidy result dict.

    Returns
    -------
    dict with keys ``"answer"`` (str) and ``"source_documents"`` (list).
    """
    result = chain({"question": question})
    return {
        "answer": result["answer"],
        "source_documents": result.get("source_documents", []),
    }
