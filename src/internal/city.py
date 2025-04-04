from langchain_core.output_parsers import StrOutputParser
from internal.prompts import create_question_type_prompt
from internal.util import create_llm
from third_party.city import get_city_data


def handle_cities(st, model_name):
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
        llm_chain = create_question_type_prompt() | create_llm(model_name) | StrOutputParser()  # noqa: E501
        with st.spinner("Loading..."):
            result = llm_chain.invoke({
                'search_query': search_query,
                'selections': selected_cities,
                'type': type
            })
        st.success(result)