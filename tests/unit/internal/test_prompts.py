"""Unit tests for internal.prompts module."""

from internal.prompts import (
    create_agentic_react_prompt,
    create_deep_agents_system_prompt,
    create_mtg_prompt,
    create_question_type_prompt,
    create_summarize_prompt,
    create_summarize_prompt_v2,
)


class TestCreateQuestionTypePrompt:
    """Tests for create_question_type_prompt function."""

    def test_prompt_created(self):
        """Test that prompt is created successfully."""
        prompt = create_question_type_prompt()

        assert prompt is not None
        assert hasattr(prompt, 'template')
        assert hasattr(prompt, 'input_variables')

    def test_prompt_input_variables(self):
        """Test that prompt has correct input variables."""
        prompt = create_question_type_prompt()

        expected_vars = {"search_query", "selections", "type"}
        assert set(prompt.input_variables) == expected_vars

    def test_prompt_template_contains_placeholders(self):
        """Test that template contains expected placeholders."""
        prompt = create_question_type_prompt()

        assert "{type}" in prompt.template
        assert "{selections}" in prompt.template
        assert "{search_query}" in prompt.template

    def test_prompt_formatting(self):
        """Test that prompt can be formatted with values."""
        prompt = create_question_type_prompt()

        formatted = prompt.format(
            type="cities",
            selections="New York, London, Tokyo",
            search_query="Which is the largest?"
        )

        assert "New York" in formatted
        assert "London" in formatted
        assert "largest" in formatted


class TestCreateMTGPrompt:
    """Tests for create_mtg_prompt function."""

    def test_prompt_created(self):
        """Test that MTG prompt is created successfully."""
        prompt = create_mtg_prompt()

        assert prompt is not None
        assert hasattr(prompt, 'template')

    def test_prompt_input_variables(self):
        """Test that MTG prompt has correct input variable."""
        prompt = create_mtg_prompt()

        assert "information" in prompt.input_variables
        assert len(prompt.input_variables) == 1

    def test_prompt_formatting(self):
        """Test that MTG prompt can be formatted."""
        prompt = create_mtg_prompt()

        formatted = prompt.format(information="Black Lotus - 0 mana artifact")

        assert "Black Lotus" in formatted
        assert "mana" in formatted

    def test_prompt_mentions_cards(self):
        """Test that prompt mentions Magic cards."""
        prompt = create_mtg_prompt()

        assert "Magic" in prompt.template or "card" in prompt.template.lower()


class TestCreateSummarizePrompt:
    """Tests for create_summarize_prompt function."""

    def test_prompt_created(self):
        """Test that summarize prompt is created successfully."""
        prompt = create_summarize_prompt()

        assert prompt is not None
        assert hasattr(prompt, 'template')

    def test_prompt_input_variables(self):
        """Test that summarize prompt has correct input variables."""
        prompt = create_summarize_prompt()

        expected_vars = {"context", "question"}
        assert set(prompt.input_variables) == expected_vars

    def test_prompt_formatting(self):
        """Test that summarize prompt can be formatted."""
        prompt = create_summarize_prompt()

        formatted = prompt.format(
            context="The Earth is round",
            question="Is the Earth flat?"
        )

        assert "The Earth is round" in formatted
        assert "flat" in formatted

    def test_prompt_requests_bullet_points(self):
        """Test that prompt asks for bullet points."""
        prompt = create_summarize_prompt()

        assert "bullet" in prompt.template.lower()

    def test_prompt_specifies_word_limits(self):
        """Test that prompt specifies word limits."""
        prompt = create_summarize_prompt()

        assert "100" in prompt.template or "200" in prompt.template


class TestCreateSummarizePromptV2:
    """Tests for create_summarize_prompt_v2 function."""

    def test_prompt_created(self):
        """Test that v2 summarize prompt is created successfully."""
        prompt = create_summarize_prompt_v2()

        assert prompt is not None
        assert hasattr(prompt, 'template')

    def test_prompt_input_variables(self):
        """Test that v2 prompt has correct input variables."""
        prompt = create_summarize_prompt_v2()

        expected_vars = {"context", "question"}
        assert set(prompt.input_variables) == expected_vars

    def test_prompt_formatting(self):
        """Test that v2 prompt can be formatted."""
        prompt = create_summarize_prompt_v2()

        formatted = prompt.format(
            context="Important information",
            question="What is important?"
        )

        # Should handle the special template format
        assert formatted is not None
        assert len(formatted) > 0

    def test_prompt_is_llama_format(self):
        """Test that v2 uses Llama chat format."""
        prompt = create_summarize_prompt_v2()

        # Check for Llama format markers
        assert "<|begin_of_text|>" in prompt.template or "<|start_header_id|>" in prompt.template  # noqa: E501

    def test_prompt_has_system_role(self):
        """Test that v2 has system role in template."""
        prompt = create_summarize_prompt_v2()

        assert "system" in prompt.template or "assistant" in prompt.template


class TestCreateAgenticReactPrompt:
    """Tests for create_agentic_react_prompt function."""

    def test_prompt_created(self):
        """Test that ReAct prompt is created successfully."""
        prompt = create_agentic_react_prompt()

        assert prompt is not None
        assert hasattr(prompt, 'template')

    def test_prompt_input_variables(self):
        """Test that ReAct prompt has required input variables."""
        prompt = create_agentic_react_prompt()

        required_vars = {"input", "agent_scratchpad", "tools", "tool_names"}
        assert all(var in prompt.input_variables for var in required_vars)

    def test_prompt_is_partial(self):
        """Test that prompt has instructions partial."""
        prompt = create_agentic_react_prompt()

        # Should have instructions in template
        assert "instructions" in str(prompt.template) or "Pokemon" in str(prompt.template)  # noqa: E501

    def test_prompt_react_format(self):
        """Test that prompt uses ReAct format."""
        prompt = create_agentic_react_prompt()

        template = prompt.template
        # Should have ReAct pattern markers
        assert "Thought:" in template
        assert "Action:" in template
        assert "Observation:" in template
        assert "Final Answer:" in template

    def test_prompt_formatting(self):
        """Test that ReAct prompt can be formatted."""
        prompt = create_agentic_react_prompt()

        formatted = prompt.format(
            input="Tell me about Pikachu",
            agent_scratchpad="",
            tools="getPokemon, getType",
            tool_names="getPokemon, getType"
        )

        assert "Pikachu" in formatted
        assert "getPokemon" in formatted

    def test_prompt_mentions_pokemon(self):
        """Test that prompt is tailored for Pokemon."""
        prompt = create_agentic_react_prompt()

        template_str = str(prompt.template)
        assert "Pokemon" in template_str or "Pikachu" in template_str

    def test_prompt_has_examples(self):
        """Test that prompt includes examples."""
        prompt = create_agentic_react_prompt()

        template_str = prompt.template
        assert "Example" in template_str or "Pikachu" in template_str

    def test_prompt_emphasizes_rules(self):
        """Test that prompt emphasizes strict rules."""
        prompt = create_agentic_react_prompt()

        template_str = prompt.template
        assert "STRICT" in template_str or "RULES" in template_str


class TestCreateDeepAgentsSystemPrompt:
    """Tests for create_deep_agents_system_prompt function."""

    def test_prompt_created(self):
        """Test that Deep Agents system prompt is created successfully."""
        prompt = create_deep_agents_system_prompt()

        assert prompt is not None
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_prompt_is_string(self):
        """Test that function returns a string (not a PromptTemplate)."""
        prompt = create_deep_agents_system_prompt()

        assert isinstance(prompt, str)

    def test_prompt_mentions_tools(self):
        """Test that prompt mentions available tools."""
        prompt = create_deep_agents_system_prompt()

        # Should mention key tools
        assert "Wikipedia" in prompt or "wikipedia" in prompt
        assert "arXiv" in prompt or "arxiv" in prompt

    def test_prompt_instructs_step_by_step_thinking(self):
        """Test that prompt encourages step-by-step thinking."""
        prompt = create_deep_agents_system_prompt()

        assert "step" in prompt.lower() or "Think" in prompt or "plan" in prompt.lower()  # noqa: E501

    def test_prompt_emphasizes_citations(self):
        """Test that prompt emphasizes citing sources."""
        prompt = create_deep_agents_system_prompt()

        assert "cite" in prompt.lower() or "source" in prompt.lower()

    def test_prompt_handles_conflict(self):
        """Test that prompt addresses handling conflicting information."""
        prompt = create_deep_agents_system_prompt()

        assert "conflict" in prompt.lower() or "disagree" in prompt.lower()

    def test_prompt_is_comprehensive(self):
        """Test that prompt is sufficiently comprehensive."""
        prompt = create_deep_agents_system_prompt()

        # Should be substantial (> 200 chars)
        assert len(prompt) > 200

    def test_prompt_can_be_used_as_system_message(self):
        """Test that prompt is suitable as a system message."""
        prompt = create_deep_agents_system_prompt()

        # Should contain role definition
        assert "assistant" in prompt.lower() or "you are" in prompt.lower()
