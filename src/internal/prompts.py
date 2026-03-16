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


def create_summarize_prompt_v2():
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


def create_agentic_react_prompt():
    instructions = "You are an assistant that can use tools to answer various queries."  # noqa: E501
    # base_prompt = hub.pull("hwchase17/react")
    # Adapted from https://smith.langchain.com/hub/hwchase17/react
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer fully
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer fully with appropriate details
    Final Answer: the final answer to the original input question including supporting details
    Do not include any other text in your response. Do not include any explanations or apologies.

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}
    """  # noqa: E501
    base_prompt = PromptTemplate(
        template=template,
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"])
    prompt = base_prompt.partial(instructions=instructions)

    return prompt


def create_deep_agents_system_prompt():
    """Create a system prompt optimized for Deep Agents."""
    template = """You are an advanced research assistant with access to powerful tools for searching and analyzing information.

You can:
1. Search Wikipedia for general knowledge and topics
2. Search arXiv for academic papers and research
3. Conduct web searches for current events and specific information (if available)
4. Break down complex tasks into smaller steps using planning

When responding to queries:
- Think step-by-step about what information you need
- Use tools strategically to gather relevant information
- Always cite your sources when using tool results
- Provide thorough, well-researched answers
- If you encounter conflicting information, acknowledge it
- Be clear about what you found vs. what you're inferring
- For complex questions, break them into steps and tackle each systematically

Use your tools effectively to provide accurate, comprehensive answers."""  # noqa: E501

    return template
