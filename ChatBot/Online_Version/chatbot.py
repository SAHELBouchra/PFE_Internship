import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from retrieval import get_retriever  # <-- import the retriever from retrieval.py



warnings.filterwarnings("ignore")

# ========== SETTINGS ==========
GOOGLE_API_KEY = "YOUR API KEY"
MODEL = "models/gemini-1.5-flash"
# ===============================

def initialize_llm(model_name: str, google_api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )

def setup_qa_chain(llm, retriever, custom_prompt):
    prompt = PromptTemplate.from_template(custom_prompt)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )

def main():
    print("✅ Initializing chatbot...")

    retriever = get_retriever(k=2)
    llm = initialize_llm(MODEL, GOOGLE_API_KEY)

    custom_prompt = """
You are an AI assistant that provides answers strictly based on the retrieved context.

    If the user greets you (like 'hello', 'hi', 'hey'), respond warmly and ask how you can help.
    
    If the user thanks you (like 'thank you' or 'thanks'), respond kindly and offer further help.

    If the information is not found in the provided context, respond with:

    "This information is not available at the moment." Do not attempt to generate an answer beyond the given context.
    Keep responses concise and to the point.

    Context: {context}
    Question: {question}
    Answer:
    """

    qa_chain = setup_qa_chain(llm, retriever, custom_prompt)

    print("\n✅ Chatbot is ready! Type your questions below.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break
        
        response = qa_chain.invoke({"query": user_input})
        
        answer = response.get("result", "⚠️ No answer returned.")
        
        print(f"AI: {answer}\n")

if __name__ == "__main__":
    main()
