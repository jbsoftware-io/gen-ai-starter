from langchain_core.output_parsers import StrOutputParser

from internal.prompts import create_question_type_prompt
from internal.util import create_llm
from third_party.country import get_country_data


def handle_country(st, model_name, langfuse_handler=None):
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
        llm_chain = create_question_type_prompt() | create_llm(model_name) | StrOutputParser()  # noqa: E501
        with st.spinner("Loading..."):
            config = {
                "callbacks": [langfuse_handler] if langfuse_handler else None,
            }
            result = llm_chain.invoke({
                'search_query': search_query,
                'selections': selected_countries,
                'type': 'Countries'
            }, config=config)
        st.success(result)
