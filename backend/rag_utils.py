
from langchain_openai import ChatOpenAI
import os

# Retrieve the top-k relevant chunks from the vector store
def retrieve(query: str, k: int = 2,vectorstore=None) -> list[str]:
    results = vectorstore.similarity_search(query, k=k)

    # Return each chunk with its content and metadata
    return [
        f"Content:\n{r.page_content}\n\nMetadata:\n{r.metadata}"
        for r in results
    ]

# Define the generation function that prompts the LLM with retrieved context
def generate(query, retrieved_chunks,model) -> str:

    prompt = f"""
    You are an AI assistant helping AuroraStay front-desk staff answer guest questions about hotel policies.

    User Query:
    {query}

    Retrieved Policy Information:
    {retrieved_chunks}

    Rules:
    1. Answer using ONLY the retrieved policy information. Do not use outside knowledge or general hotel industry practice.
    2. Answer the guest's actual question first, in the first sentence, then add supporting details.
    3. Include every condition, deadline, fee amount, limit, and exception from the retrieved information that applies to the question.
    4. If a rule is property-specific, name the property (for example, ASH-CHI Loop). Never apply one property's rule or fee to another property.
    5. If a table row appears without its column headers, do not guess which value belongs to which column. Only state values whose meaning is clear from the retrieved text.
    6. If the retrieved information only partly answers the question, answer the part that is supported and clearly state what is not covered.
    7. Use this exact reply ONLY if none of the retrieved information is relevant to the question: "The available policy documents do not contain enough information to answer this question. Please check with a supervisor." If any retrieved information is relevant, follow rule 6 instead and give a partial answer.
    8. If two retrieved passages conflict, mention both and say staff should confirm with a supervisor.
    9. Do not invent fees, times, room types, refund amounts, or policies.
    10. End with the source document name(s) the answer is based on, taken from the Metadata "source" field.
    11. Keep the answer concise, clear, and written so front-desk staff can relay it to a guest.

    Answer:
    """

    return model.invoke(prompt)

# Define the full RAG pipeline combining retrieval and generation
def rag(
    query: str,
    k: int = 2,
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 512,
    vectorstore=None
):
    # Create the generator model
    model = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        openai_api_base=os.getenv('OPENAI_API_BASE')
    )

    # Retrieve relevant chunks
    retrieved_chunks = retrieve(query=query, k=k,vectorstore=vectorstore)

    # Generate answer using retrieved chunks
    answer = generate(
        query=query,
        retrieved_chunks=retrieved_chunks,
        model=model
    )

    return answer.content, retrieved_chunks
