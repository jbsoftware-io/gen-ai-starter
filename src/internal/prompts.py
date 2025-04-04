from langchain_core.prompts import PromptTemplate


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


def create_summaryize_prompt_v2():
    template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a helpful assistant designed to help users navigate a complex set of documents. Answer the user's query based on the following context. Follow these rules:
    Use only information from the provided context.
    If the context doesn't adequately address the query, say: "Based on the available information, I cannot provide a complete answer to this question."
    Give clear, concise, and accurate responses. Explain complex terms if needed.
    If the context contains conflicting information, point this out without attempting to resolve the conflict.
    Don't use phrases like "according to the context," "as the context states," etc.
    Remember, your purpose is to provide information based on the retrieved context, not to offer original advice.
    Context: ${context}<|eot_id|><|start_header_id|>user<|end_header_id|>
    ${question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """  # noqa: E501
    prompt = PromptTemplate(
        template=template, input_variables=["context", "question"])

    return prompt
