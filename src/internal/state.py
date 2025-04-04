from langchain_core.output_parsers import StrOutputParser
from internal.prompts import create_question_type_prompt
from internal.util import create_llm
from third_party.state import get_state_data


def handle_states(st, model_name):
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
        llm_chain = create_question_type_prompt() | create_llm(model_name) | StrOutputParser()  # noqa: E501
        with st.spinner("Loading..."):
            result = llm_chain.invoke({
                'search_query': search_query,
                'selections': selected_states,
                'type': type
            })
        st.success(result)