from unittest.mock import Mock, patch, call

# Import the modules we're testing
from example.country import handle_country


class TestCountries:
    """Test cases for countries functionality"""

    def test_handle_country_no_selection(self, mock_streamlit):
        """Test country handler when no countries are selected"""

        # Mock no countries selected
        mock_streamlit.multiselect.return_value = []
        mock_streamlit.text_input.return_value = "Compare the geography"
        mock_streamlit.button.return_value = False

        with patch('example.country.get_country_data') as mock_get_country_data:  # noqa: E501
            # Mock country data
            mock_countries = {
                "United States": {"flags": {"png": "us_flag.png"}},
                "United Kingdom": {"flags": {"png": "uk_flag.png"}},
                "Japan": {"flags": {"png": "jp_flag.png"}}
            }
            mock_get_country_data.return_value = mock_countries

            handle_country(mock_streamlit, "test_model")

            # Verify UI components were called
            mock_streamlit.multiselect.assert_called_once_with(
                "Select one or more Countries",
                sorted(mock_countries.keys())
            )
            mock_streamlit.text_input.assert_called_once_with(
                "Question or Request",
                placeholder="Compare the (geography|government|economy|transportation|culture|history) of these countries."  # noqa: E501
            )
            mock_streamlit.button.assert_called_once_with("Get Information")

            # Should not show flags or process when no countries selected
            mock_streamlit.image.assert_not_called()
            mock_streamlit.success.assert_not_called()

    def test_handle_country_with_selection_and_query(self, mock_streamlit):
        """Test country handler with countries selected and query provided"""

        # Mock user selections
        selected_countries = ["United States", "Japan"]
        mock_streamlit.multiselect.return_value = selected_countries
        mock_streamlit.text_input.return_value = "Compare the economy of these countries"  # noqa: E501
        mock_streamlit.button.return_value = True

        # Mock external dependencies
        with patch('example.country.get_country_data') as mock_get_country_data:  # noqa: E501
            with patch('example.country.create_llm') as mock_create_llm:
                with patch('example.country.StrOutputParser') as mock_parser:  # noqa: E501
                    with patch('example.country.create_question_type_prompt') as mock_prompt:  # noqa: E501

                        # Mock country data
                        mock_countries = {
                            "United States": {"flags": {"png": "us_flag.png"}},  # noqa: E501
                            "United Kingdom": {"flags": {"png": "uk_flag.png"}},  # noqa: E501
                            "Japan": {"flags": {"png": "jp_flag.png"}}
                        }
                        mock_get_country_data.return_value = mock_countries

                        # Mock the final chain
                        mock_chain = Mock()
                        mock_result = "The US has a diverse economy while Japan is known for technology..."  # noqa: E501
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

                        handle_country(mock_streamlit, "test_model")

                        # Verify external dependencies were called
                        mock_get_country_data.assert_called_once()
                        mock_prompt.assert_called_once()
                        mock_create_llm.assert_called_once_with("test_model")
                        mock_parser.assert_called_once()

                        # Verify flags were displayed for selected countries
                        expected_image_calls = [
                            call("us_flag.png", width=100, caption="United States"),  # noqa: E501
                            call("jp_flag.png", width=100, caption="Japan")
                        ]
                        assert mock_streamlit.image.call_count == 2
                        assert all(call in mock_streamlit.image.call_args_list for call in expected_image_calls)  # noqa: E501

                        # Verify chain was invoked with correct parameters
                        mock_chain.invoke.assert_called_once_with({
                            'search_query': 'Compare the economy of these countries',  # noqa: E501
                            'selections': selected_countries,
                            'type': 'Countries'
                        })

                        # Verify result was displayed
                        mock_streamlit.success.assert_called_once_with(mock_result)  # noqa: E501

    def test_handle_country_shows_flags(self, mock_streamlit):
        """Test country handler shows flags for selected countries"""

        # Mock country selection
        selected_countries = ["Canada", "Brazil"]
        mock_streamlit.multiselect.return_value = selected_countries
        mock_streamlit.text_input.return_value = "Compare these countries"
        mock_streamlit.button.return_value = False

        with patch('example.country.get_country_data') as mock_get_country_data:  # noqa: E501
            # Mock country data
            mock_countries = {
                "Canada": {"flags": {"png": "ca_flag.png"}},
                "Brazil": {"flags": {"png": "br_flag.png"}}
            }
            mock_get_country_data.return_value = mock_countries

            handle_country(mock_streamlit, "test_model")

            # Verify flags were displayed
            assert mock_streamlit.image.call_count == 2
            # Check that image was called with correct parameters for each country  # noqa: E501
            image_calls = mock_streamlit.image.call_args_list
            assert any("ca_flag.png" in str(call) and "Canada" in str(call) for call in image_calls)  # noqa: E501
            assert any("br_flag.png" in str(call) and "Brazil" in str(call) for call in image_calls)  # noqa: E501

    def test_handle_country_no_button_click(self, mock_streamlit):
        """Test country handler when button is not clicked"""

        # Mock selections but no button click
        mock_streamlit.multiselect.return_value = ["Germany", "France"]
        mock_streamlit.text_input.return_value = "Compare the culture"
        mock_streamlit.button.return_value = False

        with patch('example.country.get_country_data') as mock_get_country_data:  # noqa: E501
            # Mock country data
            mock_countries = {
                "Germany": {"flags": {"png": "de_flag.png"}},
                "France": {"flags": {"png": "fr_flag.png"}}
            }
            mock_get_country_data.return_value = mock_countries

            handle_country(mock_streamlit, "test_model")

            # Verify UI components were called but no processing happened
            mock_get_country_data.assert_called_once()
            mock_streamlit.multiselect.assert_called_once()
            mock_streamlit.text_input.assert_called_once()
            mock_streamlit.button.assert_called_once()

            # Should show flags but not success message when button not clicked  # noqa: E501
            assert mock_streamlit.image.call_count == 2
            mock_streamlit.success.assert_not_called()
