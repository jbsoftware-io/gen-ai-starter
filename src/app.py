from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb, os, tempfile, streamlit as st  # noqa: E401
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from langchain_postgres import PGVector
from third_party.city import get_city_data
from third_party.country import get_country_data
from third_party.mtg import get_card_data
from third_party.state import get_state_data

st.title("Generative AI Demo")

with st.sidebar:
    type = st.selectbox(
        "Select a Type",
        ["Countries", "States", "Cities", "MTG", "Chroma", "PG_Vector"]
    )
    model_path = st.text_input(
        "Model Path",
        "/Users/jbilyeu/Library/Application Support/nomic.ai/GPT4All"
    )
    # get all .gguf files under model_path
    model_files = [
        f for f in os.listdir(model_path) if f.endswith(".gguf")
    ] if model_path else []
    model_name = st.selectbox("Model Name", model_files)


def create_question_type_prompt():
    template = """Question: We are in the context of these {type}: {selections}.  Without defining the question, please concisely answer the following question or request: "{search_query}"

    Answer: Please be concise and summarize in a few sentences."""  # noqa: E501

    prompt = PromptTemplate(
        template=template,
        input_variables=["search_query", "selections", "type"]
    )

    return prompt


def create_mtg_prompt():
    template = """
    Given the following Magic the Gather card information: "{information}", I want you to create just the following two items:
    * A short summary of most important info, no more than 200 words
    * two interesting facts about this card
    """  # noqa: E501

    prompt = PromptTemplate(template=template, input_variables=["information"])

    return prompt


def create_summarize_prompt():
    template = """
    Context: {context}
    Question: {question}
    Instructions: I want you to create only the following two items in response:
    * 2-3 bullet points with reasons for each, no more than 100 words each reason
    * Final Summary: A brief final summary, no more than 200 words

    Answer: Please think and be thoughtful and concise in your response.
    """  # noqa: E501

    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"])

    return prompt


def create_llm():
    if not model_path or not model_name:
        st.error("Please provide a valid model path and model name.")
        return
    local_path = f"{model_path}/{model_name}"
    print(f"Using model: {local_path}")

    # callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # verbose is required to pass to the callback manager
    llm = GPT4All(
        model=local_path,
        callbacks=callbacks,
        max_tokens=4000,
        verbose=True
    )
    return llm


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if type == "Countries":
    countries_map_by_common_name = get_country_data()

    # pick one or more countries
    selected_countries = st.multiselect(
        "Select one or more Countries",
        sorted(countries_map_by_common_name.keys())
    )
    # question about the countries
    search_query = st.text_input(
        "Question or Request",
        placeholder="Compare the (geography|government|economy|transportation|culture|history) of these countries."  # noqa: E501
    )

    # show the flags of the selected countries
    for country in selected_countries:
        st.image(
            countries_map_by_common_name[country]["flags"]["png"],
            width=100,
            caption=country
        )

    if st.button("Get Information"):
        llm_chain = create_question_type_prompt() | create_llm() | StrOutputParser()  # noqa: E501
        with st.spinner("Loading..."):
            result = llm_chain.invoke({
                'search_query': search_query,
                'selections': selected_countries,
                'type': type
            })
        st.success(result)

elif type == "States":
    sorted_states = get_state_data()

    # pick one or more states
    selected_states = st.multiselect(
        "Select one or more States",
        sorted_states
    )
    # question about the states
    search_query = st.text_input(
        "Question or Request",
        placeholder="Compare the (geography|government|economy|transportation|culture|history) of these states."  # noqa: E501
    )

    if st.button("Get Information"):
        llm_chain = create_question_type_prompt() | create_llm() | StrOutputParser()  # noqa: E501
        with st.spinner("Loading..."):
            result = llm_chain.invoke({
                'search_query': search_query,
                'selections': selected_states,
                'type': type
            })
        st.success(result)

elif type == "Cities":
    sorted_cities = get_city_data()

    # pick one or more cities
    selected_cities = st.multiselect(
        "Select one or more Cities",
        sorted_cities
    )

    # question about the cities
    search_query = st.text_input(
        "Question or Request",
        placeholder="Compare the (geography|government|economy|transportation|culture|history) of these cities."  # noqa: E501
    )

    if st.button("Get Information"):
        llm_chain = create_question_type_prompt() | create_llm() | StrOutputParser()  # noqa: E501
        with st.spinner("Loading..."):
            result = llm_chain.invoke({
                'search_query': search_query,
                'selections': selected_cities,
                'type': type
            })
        st.success(result)

elif type == "MTG":
    card_name = st.text_input(
        "Card Name",
        placeholder="Enter a Magic the Gathering card name (leave blank for random card)"  # noqa: E501
    )

    if st.button("Get Information"):
        llm_chain = create_mtg_prompt() | create_llm() | StrOutputParser()
        card_data, card = get_card_data(card_name)

        # display card image in middle column (fall back to name if no image)
        _, col2, _ = st.columns(3)
        if card['image_url']:
            with col2:
                st.image(card['image_url'], width=300, caption=card['name'])
        else:
            with col2:
                st.write(card['name'])

        with st.spinner("Loading..."):
            # invoke request to LLM with the JSON card data
            result = llm_chain.invoke({'information': card_data})

        st.success(result)

elif type == "Chroma":
    chroma_client = chromadb.HttpClient(
        host="localhost",
        port=8000,
        settings=Settings(allow_reset=True, anonymized_telemetry=False))
    # chroma_client.reset()  # resets the database
    source_doc = st.file_uploader(
        "Source PDF Document",
        label_visibility="collapsed",
        type="pdf"
    )
    search_query = st.text_input(
        "Question",
        placeholder="Ask a question about the uploaded document."
    )

    if st.button("Summarize"):
        with st.spinner('Please wait...'):
            if source_doc:
                try:
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, source_doc.name)
                    with open(path, "wb") as f:
                        f.write(source_doc.getvalue())

                    loader = PyPDFLoader(
                        file_path=path
                    )

                    data = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=0
                    )
                    all_splits = text_splitter.split_documents(data)

                    collection_name = os.path.basename(path)

                    collection = chroma_client.create_collection(
                        collection_name, get_or_create=True)

                    embeddings = GPT4AllEmbeddings(
                        model_path=model_path,
                        model_name=model_name
                    )

                    # Open from documents
                    # vectorstore = Chroma.from_documents(
                    #     documents=all_splits,
                    #     embedding=embeddings
                    # )

                    # tell LangChain to use our client and collection name
                    vectorstore = Chroma(
                        client=chroma_client,
                        collection_name=collection_name,
                        embedding_function=embeddings,
                    )

                    # if the collection is empty, add the documents again
                    if collection.count() == 0:
                        print("Adding documents")
                        vectorstore.add_documents(all_splits)

                    llm = create_llm()

                    summarize_prompt = create_summarize_prompt()

                    rag_chain_from_docs = (
                        RunnablePassthrough.assign(
                            context=(lambda x: format_docs(x["context"])))
                        | summarize_prompt
                        | llm
                        | StrOutputParser()
                    )

                    retrieve_docs = (
                        lambda x: x["question"]
                    ) | vectorstore.as_retriever()

                    print("Invoking chain")
                    chain = RunnablePassthrough.assign(
                        context=retrieve_docs
                    ).assign(answer=rag_chain_from_docs)

                    result = chain.invoke({"question": search_query})

                    print("Result")
                    print(result)
                    print('-'*30)

                    if not result or not result['answer']:
                        st.warning("No answer was found.")
                    else:
                        st.success(result['answer'])

                except Exception as e:
                    st.exception(f"An error occurred: {e}")
elif type == "PG_Vector":
    connection = "postgresql+psycopg://myuser:ChangeMe@localhost:5432/api"  # noqa: E501

    source_doc = st.file_uploader(
        "Source PDF Document",
        label_visibility="collapsed",
        type="pdf"
    )
    search_query = st.text_input(
        "Question",
        placeholder="Ask a question about the uploaded document."
    )

    if st.button("Summarize"):
        with st.spinner('Please wait...'):
            if source_doc:
                try:
                    temp_dir = tempfile.mkdtemp()
                    path = os.path.join(temp_dir, source_doc.name)
                    with open(path, "wb") as f:
                        f.write(source_doc.getvalue())

                    loader = PyPDFLoader(
                        file_path=path
                    )

                    data = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=0
                    )
                    all_splits = text_splitter.split_documents(data)

                    col_name = os.path.basename(path)

                    embeddings = GPT4AllEmbeddings(
                        model_path=model_path,
                        model_name=model_name
                    )

                    general_store = PGVector(
                        embeddings=embeddings,
                        connection=connection,
                        use_jsonb=True,
                    )

                    with general_store.session_maker() as session:
                        # if the collection is empty, add the documents again
                        collection_store = general_store.get_collection(session)
                        collection, created = collection_store.get_or_create(
                            session, col_name)

                        print(f"Collection {col_name} created: {created}")
                        
                        vector_store = PGVector(
                            embeddings=embeddings,
                            connection=connection,
                            collection_name=col_name,
                            use_jsonb=True,
                        )
                        if created:
                            print("Adding documents")
                            vector_store.add_documents(all_splits)

                    llm = create_llm()

                    summarize_prompt = create_summarize_prompt()

                    rag_chain_from_docs = (
                        RunnablePassthrough.assign(
                            context=(lambda x: format_docs(x["context"])))
                        | summarize_prompt
                        | llm
                        | StrOutputParser()
                    )

                    retrieve_docs = (
                        lambda x: x["question"]
                    ) | vector_store.as_retriever()

                    print("Invoking chain")
                    chain = RunnablePassthrough.assign(
                        context=retrieve_docs
                    ).assign(answer=rag_chain_from_docs)

                    result = chain.invoke({"question": search_query})

                    print("Result")
                    print(result)
                    print('-'*30)

                    if not result or not result['answer']:
                        st.warning("No answer was found.")
                    else:
                        st.success(result['answer'])

                except Exception as e:
                    st.exception(f"An error occurred: {e}")
