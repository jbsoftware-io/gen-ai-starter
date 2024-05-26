from langchain_text_splitters import RecursiveCharacterTextSplitter
import os, tempfile, streamlit as st  # noqa: E401
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings.gpt4all import GPT4AllEmbeddings
from langchain_community.llms import GPT4All
from langchain_core.prompts import PromptTemplate
from third_party.city import get_city_data
from third_party.country import get_country_data
from third_party.mtg import get_card_data
from third_party.state import get_state_data


st.title("Generative AI Demo")

# Get OpenAI API key, Serper API key, number of results, and search query
with st.sidebar:
    type = st.selectbox(
        "Select a Type",
        ["Countries", "States", "Cities", "MTG", "Chroma"]
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


if type == "Countries":
    countries_map_by_common_name = get_country_data()

    # pick one or more countries
    selected_countries = st.multiselect(
        "Select one or more Contries",
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
    source_doc = st.file_uploader(
        "Source Document (must have 'Sentences' column)",
        label_visibility="collapsed",
        type="csv"
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

                    loader = CSVLoader(
                        file_path=path,
                        source_column="Sentences"
                    )

                    data = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=0
                    )
                    all_splits = text_splitter.split_documents(data)

                    embeddings = GPT4AllEmbeddings(
                        model_path=model_path,
                        model_name=model_name
                    )
                    vectorstore = Chroma.from_documents(
                        documents=all_splits,
                        embedding=embeddings
                    )

                    llm = create_llm()
                    chain = load_summarize_chain(llm, chain_type="stuff")
                    search = vectorstore.similarity_search(search_query)
                    summary = chain.run(
                        input_documents=search,
                        question="Write a summary within 200 words.",
                        max_tokens=4000
                    )

                    st.success(summary)
                except Exception as e:
                    st.exception(f"An error occurred: {e}")
