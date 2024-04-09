#streamlit run app.py
import os
from pathlib import Path
import streamlit as st

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import SKLearnVectorStore
from dotenv import load_dotenv

from load_method import load_files

with st.sidebar:
    st.title("Useful Information")
    st.markdown('''
    - This is the SSG Chatbot 0.1 in order to better understand how it works
    - We use gpt-4-turbo and text-embedding-ada-002 of Azure OpenAI due to low cost
    - Write the index name
        - If it does not already exist you can select the pdfs that you want to use for chatting
        - Else it will load the embeddings from disk and use the pdfs that you have chosen the previous time
    - Any issues please contact your best friend: shaun.xu@parthenon.ey.com'''
    )

def main():
    # Display app header
    st.header("SSG - Chat with PDF 🤖💬")

    # Check if 'messages' key exists in st.session_state, if not, initialize it
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Placeholder for uploaded PDFs
    pdfs = None

    # List to store existing conversation names
    runs = []
    for file in os.listdir():
        if file.endswith('.json'):
            runs.append(file[:-12])

    # Display existing conversation names
    st.write(runs)

    # Input for conversation name
    index_name = st.text_input(
        "Enter the name of the conversation",
        label_visibility="visible",
        placeholder="Write text here",
        value = None
    )

    if index_name not in runs and index_name is not None:
        # upload multiple pdf files
        # pdfs = st.file_uploader("Upload your PDFs", type='pdf', accept_multiple_files=True)
        pdfs = st.file_uploader("Upload your documents", type='pdf', accept_multiple_files=True)

    # Check if conversation name exists or PDFs are uploaded
    if pdfs or index_name in runs:

        # Initialize embedding model
        embed_model = AzureOpenAIEmbeddings(
            azure_deployment=os.environ.get('EMBEDDING_NAME'),
            openai_api_version="your Azure_Open_AI_API_info",
        )

        # Load existing embeddings if conversation name exists
        if index_name in runs:
            vectorstore = SKLearnVectorStore(embed_model,
                                              persist_path=f"{index_name}vctrstr.json",
                                              serializer='json'
                                              )

            st.write("Embeddings loaded from disk")
        else:
            # Load files and create a vectorstore with the embeddings
            data, vectorstore = load_files(embed_model, pdfs, index_name)
            st.write("Files Loaded")


        # Initialize RAG model
        llm = AzureChatOpenAI(
            deployment_name=os.environ.get('LLM3_NAME'),
            model_name=os.environ.get('LLM3_MODEL')
        )

        # Initialize QA chain
        chain = load_qa_chain(llm=llm, chain_type='stuff')


        # Display existing chat history
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Get user input
        if query := st.chat_input():
            # Perform similarity search to get relevant documents
            if len(vectorstore._texts) < 4:
                k_vec = len(vectorstore._texts)

            else:
            #get 4 chunks of document in order to generate an answer
                k_vec = 4

            #vectorstore.similarity_search_with_relevance_scores
            docs = vectorstore.similarity_search(query=query, k=k_vec)
            st.session_state.messages.append({"role": "user", "content": query})
            st.chat_message("user").write(query)

            # Get response and cost of each query using QA chain
            with get_openai_callback() as cb:
                res = chain.run(input_documents=docs, question=query)
                print(cb)

            # Update chat history with assistant's response
            st.session_state.messages.append({"role": "assistant", "content": res})
            st.chat_message("assistant").write(res)

if __name__ == '__main__':
    # Set the base directory for the app
    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)
    load_dotenv(dotenv_path=r".env")

    # Run the main function
    main()
