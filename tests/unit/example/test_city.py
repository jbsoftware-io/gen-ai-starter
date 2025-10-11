from unittest.mock import Mock, patch

# Import the modules we're testing
from example.city import handle_cities


class TestCities:
    """Test cases for cities functionality"""

    def test_handle_cities_no_selection(self, mock_streamlit):
        """Test cities handler when no cities are selected"""

        # Mock no cities selected
        mock_streamlit.multiselect.return_value = []
        mock_streamlit.text_input.return_value = "Compare the geography"
        mock_streamlit.button.return_value = False

        with patch('example.city.get_city_data') as mock_get_city_data:
            # Mock city data
            mock_cities = ["New York", "London", "Tokyo"]
            mock_get_city_data.return_value = mock_cities

            handle_cities(mock_streamlit, "test_model")

            # Verify UI components were called
            mock_streamlit.multiselect.assert_called_once_with(
                "Select one or more Cities",
                mock_cities
            )
            mock_streamlit.text_input.assert_called_once_with(
                "Question or Request",
                placeholder="Compare the (geography|government|economy|transportation|culture|history) of these cities."  # noqa: E501
            )
            mock_streamlit.button.assert_called_once_with("Get Information")

            # Should not process when button not clicked
            mock_streamlit.success.assert_not_called()

    def test_handle_cities_with_selection_and_query(self, mock_streamlit):
        """Test cities handler with cities selected and query provided"""

        # Mock user selections
        selected_cities = ["New York", "London"]
        mock_streamlit.multiselect.return_value = selected_cities
        mock_streamlit.text_input.return_value = "Compare the geography of these cities"  # noqa: E501
        mock_streamlit.button.return_value = True

        # Mock the final result
        mock_result = "New York has diverse geography while London is known for..."  # noqa: E501

        # Mock external dependencies
        with patch('example.city.get_city_data') as mock_get_city_data:
            with patch('example.city.create_llm') as mock_create_llm:
                with patch('example.city.StrOutputParser') as mock_str_parser:
                    with patch('example.city.create_question_type_prompt') as mock_prompt:  # noqa: E501

                        # Setup mocks
                        mock_cities = ["New York", "London", "Tokyo"]
                        mock_get_city_data.return_value = mock_cities

                        # Mock the final chain
                        mock_chain = Mock()
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
                        mock_str_parser.return_value = mock_parser_instance

                        handle_cities(mock_streamlit, "test_model")

                        # Verify external dependencies were called
                        mock_get_city_data.assert_called_once()
                        mock_prompt.assert_called_once()
                        mock_create_llm.assert_called_once_with("test_model")
                        mock_str_parser.assert_called_once()

                        # Verify chain was invoked with correct parameters
                        mock_chain.invoke.assert_called_once_with({
                            'search_query': 'Compare the geography of these cities',  # noqa: E501
                            'selections': selected_cities,
                            'type': 'Cities'
                        })

                        # Verify result was displayed
                        mock_streamlit.success.assert_called_once_with(
                            mock_result)

    def test_handle_cities_empty_query(self, mock_streamlit):
        """Test cities handler with empty query"""

        # Mock selections with empty query
        mock_streamlit.multiselect.return_value = ["New York"]
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.button.return_value = True

        with patch('example.city.get_city_data') as mock_get_city_data:
            with patch('example.city.create_llm') as mock_create_llm:
                with patch('example.city.StrOutputParser') as mock_str_parser:
                    with patch('example.city.create_question_type_prompt') as mock_prompt:  # noqa: E501

                        # Mock city data
                        mock_cities = ["New York", "London", "Tokyo"]
                        mock_get_city_data.return_value = mock_cities

                        # Mock the final chain for empty query processing
                        mock_chain = Mock()
                        mock_result = "Empty query processed"
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
                        mock_str_parser.return_value = mock_parser_instance

                        # Since the function doesn't validate empty queries, it should still process  # noqa: E501
                        # This tests the actual behavior
                        handle_cities(mock_streamlit, "test_model")

                        # Verify basic UI calls were made
                        mock_get_city_data.assert_called_once()
                        mock_streamlit.multiselect.assert_called_once()
                        mock_streamlit.text_input.assert_called_once()
                        mock_streamlit.button.assert_called_once()

                        # Verify the chain was invoked with empty query
                        mock_chain.invoke.assert_called_once_with({
                            'search_query': '',
                            'selections': ["New York"],
                            'type': 'Cities'
                        })

                        # Verify result was displayed
                        mock_streamlit.success.assert_called_once_with(
                            mock_result)

    def test_handle_cities_no_button_click(self, mock_streamlit):
        """Test cities handler when button is not clicked"""

        # Mock selections but no button click
        mock_streamlit.multiselect.return_value = ["New York", "London"]
        mock_streamlit.text_input.return_value = "Compare the geography"
        mock_streamlit.button.return_value = False

        with patch('example.city.get_city_data') as mock_get_city_data:
            # Mock city data
            mock_cities = ["New York", "London", "Tokyo"]
            mock_get_city_data.return_value = mock_cities

            handle_cities(mock_streamlit, "test_model")

            # Verify UI components were called but no processing happened
            mock_get_city_data.assert_called_once()
            mock_streamlit.multiselect.assert_called_once()
            mock_streamlit.text_input.assert_called_once()
            mock_streamlit.button.assert_called_once()

            # Should not show success message when button not clicked
            mock_streamlit.success.assert_not_called()
