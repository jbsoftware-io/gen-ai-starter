from langchain_core.output_parsers import StrOutputParser
from internal.prompts import create_mtg_prompt
from internal.util import create_llm
from third_party.mtg import get_card_data


def handle_mtg(st, model_name):
    card_name = st.text_input(
        "Card Name",
        placeholder="Enter a Magic the Gathering card name (leave blank for random card)"  # noqa: E501
    )

    if st.button("Get Information"):
        llm_chain = create_mtg_prompt() | create_llm(model_name) | StrOutputParser()
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