from unittest.mock import Mock, patch

# Import the modules we're testing
from example.state import handle_states


class TestStates:
    """Test cases for states functionality"""

    def test_handle_states_no_selection(self, mock_streamlit):
        """Test states handler when no states are selected"""

        # Mock no states selected
        mock_streamlit.multiselect.return_value = []
        mock_streamlit.text_input.return_value = "Compare the geography"
        mock_streamlit.button.return_value = False

        with patch('example.state.get_state_data') as mock_get_state_data:
            # Mock state data
            mock_states = ["California", "Texas", "New York", "Florida"]
            mock_get_state_data.return_value = mock_states

            handle_states(mock_streamlit, "test_model")

            # Verify UI components were called
            mock_streamlit.multiselect.assert_called_once_with(
                "Select one or more States",
                mock_states
            )
            mock_streamlit.text_input.assert_called_once_with(
                "Question or Request",
                placeholder="Compare the (geography|government|economy|transportation|culture|history) of these states."  # noqa: E501
            )
            mock_streamlit.button.assert_called_once_with("Get Information")

            # Should not process when button not clicked
            mock_streamlit.success.assert_not_called()

    def test_handle_states_with_selection_and_query(self, mock_streamlit):
        """Test states handler with states selected and query provided"""

        # Mock user selections
        selected_states = ["California", "Texas"]
        mock_streamlit.multiselect.return_value = selected_states
        mock_streamlit.text_input.return_value = "Compare the economy of these states"  # noqa: E501
        mock_streamlit.button.return_value = True

        # Mock external dependencies
        with patch('example.state.get_state_data') as mock_get_state_data:
            with patch('example.state.create_llm') as mock_create_llm:
                with patch('example.state.StrOutputParser') as mock_parser:
                    with patch('example.state.create_question_type_prompt') as mock_prompt:  # noqa: E501

                        # Mock state data
                        mock_states = ["California", "Texas", "New York", "Florida"]  # noqa: E501
                        mock_get_state_data.return_value = mock_states

                        # Mock the final chain
                        mock_chain = Mock()
                        mock_result = "California has a tech-driven economy while Texas focuses on energy..."  # noqa: E501
                        mock_chain.invoke.return_value = mock_result

                        # Create mock components that support the pipe operator  # noqa: E501
                        mock_prompt_instance = Mock()
                        mock_llm = Mock()
                        mock_parser_instance = Mock()

                        # Set up the pipe operator chain: prompt | llm | parser  # noqa: E501
                        mock_prompt_instance.__or__ = Mock(return_value=Mock())  # noqa: E501
                        mock_prompt_instance.__or__.return_value.__or__ = Mock(return_value=mock_chain)  # noqa: E501

                        # Wire up the mock returns
                        mock_prompt.return_value = mock_prompt_instance
                        mock_create_llm.return_value = mock_llm
                        mock_parser.return_value = mock_parser_instance

                        handle_states(mock_streamlit, "test_model")

                        # Verify external dependencies were called
                        mock_get_state_data.assert_called_once()
                        mock_prompt.assert_called_once()
                        mock_create_llm.assert_called_once_with("test_model")
                        mock_parser.assert_called_once()

                        # Verify chain was invoked with correct parameters
                        mock_chain.invoke.assert_called_once_with({
                            'search_query': 'Compare the economy of these states',  # noqa: E501
                            'selections': selected_states,
                            'type': 'States'
                        }, config={
                            "callbacks": None
                        })

                        # Verify result was displayed
                        mock_streamlit.success.assert_called_once_with(mock_result)  # noqa: E501

    def test_handle_states_empty_query(self, mock_streamlit):
        """Test states handler with empty query"""

        # Mock selections with empty query
        mock_streamlit.multiselect.return_value = ["Alaska"]
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.button.return_value = True

        with patch('example.state.get_state_data') as mock_get_state_data:
            with patch('example.state.create_llm') as mock_create_llm:
                with patch('example.state.StrOutputParser') as mock_parser:
                    with patch('example.state.create_question_type_prompt') as mock_prompt:  # noqa: E501

                        # Mock state data
                        mock_states = ["Alaska", "Hawaii"]
                        mock_get_state_data.return_value = mock_states

                        # Mock the final chain for empty query processing
                        mock_chain = Mock()
                        mock_result = "Empty query processed"
                        mock_chain.invoke.return_value = mock_result

                        # Create mock components that support the pipe operator
                        mock_prompt_instance = Mock()
                        mock_llm = Mock()
                        mock_parser_instance = Mock()

                        # Set up the pipe operator chain: prompt | llm | parser
                        mock_prompt_instance.__or__ = Mock(return_value=Mock())
                        mock_prompt_instance.__or__.return_value.__or__ = Mock(return_value=mock_chain)  # noqa: E501

                        # Wire up the mock returns
                        mock_prompt.return_value = mock_prompt_instance
                        mock_create_llm.return_value = mock_llm
                        mock_parser.return_value = mock_parser_instance

                        # Since the function doesn't validate empty queries, it should still process  # noqa: E501
                        handle_states(mock_streamlit, "test_model")

                        # Verify basic UI calls were made
                        mock_get_state_data.assert_called_once()
                        mock_streamlit.multiselect.assert_called_once()
                        mock_streamlit.text_input.assert_called_once()
                        mock_streamlit.button.assert_called_once()

                        # Verify the chain was invoked with empty query
                        mock_chain.invoke.assert_called_once_with({
                            'search_query': '',
                            'selections': ["Alaska"],
                            'type': 'States'
                        }, config={"callbacks": None})

                        # Verify result was displayed
                        mock_streamlit.success.assert_called_once_with(mock_result)  # noqa: E501

    def test_handle_states_no_button_click(self, mock_streamlit):
        """Test states handler when button is not clicked"""

        # Mock selections but no button click
        mock_streamlit.multiselect.return_value = ["Nevada", "Oregon"]
        mock_streamlit.text_input.return_value = "Compare the geography"
        mock_streamlit.button.return_value = False

        with patch('example.state.get_state_data') as mock_get_state_data:
            # Mock state data
            mock_states = ["Nevada", "Oregon", "Washington"]
            mock_get_state_data.return_value = mock_states

            handle_states(mock_streamlit, "test_model")

            # Verify UI components were called but no processing happened
            mock_get_state_data.assert_called_once()
            mock_streamlit.multiselect.assert_called_once()
            mock_streamlit.text_input.assert_called_once()
            mock_streamlit.button.assert_called_once()

            # Should not show success message when button not clicked
            mock_streamlit.success.assert_not_called()

    def test_handle_states_multiple_selections(self, mock_streamlit):
        """Test states handler with multiple state selections"""

        # Mock multiple selections
        selected_states = ["Illinois", "Michigan", "Ohio"]
        mock_streamlit.multiselect.return_value = selected_states
        mock_streamlit.text_input.return_value = "Compare the transportation systems"  # noqa: E501
        mock_streamlit.button.return_value = True

        # Mock external dependencies
        with patch('example.state.get_state_data') as mock_get_state_data:
            with patch('example.state.create_llm') as mock_create_llm:
                with patch('example.state.StrOutputParser') as mock_parser:
                    with patch('example.state.create_question_type_prompt') as mock_prompt:  # noqa: E501

                        # Mock state data
                        mock_states = ["Illinois", "Michigan", "Ohio", "Indiana"]  # noqa: E501
                        mock_get_state_data.return_value = mock_states

                        # Mock the final chain
                        mock_chain = Mock()
                        mock_result = "These Midwest states have different transportation approaches..."  # noqa: E501
                        mock_chain.invoke.return_value = mock_result

                        # Create mock components that support the pipe operator  # noqa: E501
                        mock_prompt_instance = Mock()
                        mock_llm = Mock()
                        mock_parser_instance = Mock()

                        # Set up the pipe operator chain: prompt | llm | parser  # noqa: E501
                        mock_prompt_instance.__or__ = Mock(return_value=Mock())  # noqa: E501
                        mock_prompt_instance.__or__.return_value.__or__ = Mock(return_value=mock_chain)  # noqa: E501

                        # Wire up the mock returns
                        mock_prompt.return_value = mock_prompt_instance
                        mock_create_llm.return_value = mock_llm
                        mock_parser.return_value = mock_parser_instance

                        handle_states(mock_streamlit, "test_model")

                        # Verify all three states were passed to the chain
                        mock_chain.invoke.assert_called_once_with({
                            'search_query': 'Compare the transportation systems',  # noqa: E501
                            'selections': ["Illinois", "Michigan", "Ohio"],
                            'type': 'States'
                        }, config={
                            "callbacks": None
                        })
