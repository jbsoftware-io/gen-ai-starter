from unittest.mock import Mock, patch

# Import the modules we're testing
from example.mtg import handle_mtg


class TestMTG:
    """Test cases for Magic the Gathering functionality"""

    def test_handle_mtg_no_button_click(self, mock_streamlit):
        """Test MTG handler when button is not clicked"""

        # Mock card name input
        mock_streamlit.text_input.return_value = "Lightning Bolt"
        mock_streamlit.button.return_value = False

        handle_mtg(mock_streamlit, "test_model")

        # Verify UI components were called
        mock_streamlit.text_input.assert_called_once_with(
            "Card Name",
            placeholder="Enter a Magic the Gathering card name (leave blank for random card)"  # noqa: E501
        )
        mock_streamlit.button.assert_called_once_with("Get Information")

        # Should not get card data or success when button not clicked
        mock_streamlit.success.assert_not_called()

    def test_handle_mtg_with_card_image(self, mock_streamlit):
        """Test MTG handler with card that has an image"""

        # Mock user input
        mock_streamlit.text_input.return_value = "Lightning Bolt"
        mock_streamlit.button.return_value = True

        # Mock streamlit columns
        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
        # Make columns support context manager protocol
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        mock_streamlit.columns.return_value = (
            mock_col1, mock_col2, mock_col3)

        # Mock external dependencies
        with patch('example.mtg.get_card_data') as mock_get_card_data:
            with patch('example.mtg.create_llm') as mock_create_llm:
                with patch('example.mtg.StrOutputParser') as mock_parser:
                    with patch('example.mtg.create_mtg_prompt') as mock_prompt:  # noqa: E501

                        # Mock card data
                        mock_card_data = {
                            "name": "Lightning Bolt",
                            "mana_cost": "{R}",
                            "type": "Instant"
                        }
                        mock_card = {
                            "name": "Lightning Bolt",
                            "image_url": "https://example.com/lightning_bolt.jpg"  # noqa: E501
                        }
                        mock_get_card_data.return_value = (mock_card_data, mock_card)  # noqa: E501

                        # Mock the final chain
                        mock_chain = Mock()
                        mock_result = "Lightning Bolt is a classic red instant spell..."  # noqa: E501
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

                        handle_mtg(mock_streamlit, "test_model")

                        # Verify external dependencies were called
                        mock_get_card_data.assert_called_once_with("Lightning Bolt")  # noqa: E501
                        mock_prompt.assert_called_once()
                        mock_create_llm.assert_called_once_with("test_model")
                        mock_parser.assert_called_once()

                        # Verify UI layout
                        mock_streamlit.columns.assert_called_once_with(3)

                        # Verify image was displayed (image_url exists)
                        mock_streamlit.image.assert_called_once_with(
                            "https://example.com/lightning_bolt.jpg",
                            width=300,
                            caption="Lightning Bolt"
                        )

                        # Verify chain was invoked with correct parameters
                        mock_chain.invoke.assert_called_once_with({
                            'information': mock_card_data
                        }, config={
                            "callbacks": None
                        })

                        # Verify result was displayed
                        mock_streamlit.success.assert_called_once_with(
                            mock_result)

    def test_handle_mtg_without_card_image(self, mock_streamlit):
        """Test MTG handler with card that has no image"""

        # Mock user input
        mock_streamlit.text_input.return_value = "Ancient Card"
        mock_streamlit.button.return_value = True

        # Mock streamlit columns
        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
        # Make columns support context manager protocol
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        mock_streamlit.columns.return_value = (mock_col1, mock_col2, mock_col3)  # noqa: E501

        # Mock external dependencies
        with patch('example.mtg.get_card_data') as mock_get_card_data:
            with patch('example.mtg.create_llm') as mock_create_llm:
                with patch('example.mtg.StrOutputParser') as mock_parser:
                    with patch('example.mtg.create_mtg_prompt') as mock_prompt:  # noqa: E501

                        # Mock card data without image
                        mock_card_data = {
                            "name": "Ancient Card",
                            "mana_cost": "{3}",
                            "type": "Artifact"
                        }
                        mock_card = {
                            "name": "Ancient Card",
                            "image_url": None  # No image available
                        }
                        mock_get_card_data.return_value = (
                            mock_card_data, mock_card)

                        # Mock the final chain
                        mock_chain = Mock()
                        mock_result = "Ancient Card is a mysterious artifact..."  # noqa: E501
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

                        handle_mtg(mock_streamlit, "test_model")

                        # Verify external dependencies were called
                        mock_get_card_data.assert_called_once_with("Ancient Card")  # noqa: E501

                        # Verify UI layout
                        mock_streamlit.columns.assert_called_once_with(3)

                        # Verify no image was displayed but name was written instead  # noqa: E501
                        mock_streamlit.image.assert_not_called()
                        mock_streamlit.write.assert_called_once_with("Ancient Card")  # noqa: E501
                        mock_streamlit.success.assert_called_once_with(mock_result)  # noqa: E501

    def test_handle_mtg_empty_card_name(self, mock_streamlit):
        """Test MTG handler with empty card name (random card)"""

        # Mock empty card name input
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.button.return_value = True

        # Mock streamlit columns
        mock_col1, mock_col2, mock_col3 = Mock(), Mock(), Mock()
        # Make columns support context manager protocol
        mock_col1.__enter__ = Mock(return_value=mock_col1)
        mock_col1.__exit__ = Mock(return_value=None)
        mock_col2.__enter__ = Mock(return_value=mock_col2)
        mock_col2.__exit__ = Mock(return_value=None)
        mock_col3.__enter__ = Mock(return_value=mock_col3)
        mock_col3.__exit__ = Mock(return_value=None)
        mock_streamlit.columns.return_value = (
            mock_col1, mock_col2, mock_col3)

        # Mock external dependencies
        with patch('example.mtg.get_card_data') as mock_get_card_data:
            with patch('example.mtg.create_llm') as mock_create_llm:
                with patch('example.mtg.StrOutputParser') as mock_parser:
                    with patch('example.mtg.create_mtg_prompt') as mock_prompt:  # noqa: E501

                        # Mock random card data
                        mock_card_data = {
                            "name": "Random Card",
                            "mana_cost": "{2}",
                            "type": "Creature"
                        }
                        mock_card = {
                            "name": "Random Card",
                            "image_url": "https://example.com/random_card.jpg"
                        }
                        mock_get_card_data.return_value = (mock_card_data, mock_card)  # noqa: E501

                        # Mock the final chain
                        mock_chain = Mock()
                        mock_result = "Random Card is an interesting creature..."  # noqa: E501
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

                        handle_mtg(mock_streamlit, "test_model")

                        # Verify get_card_data was called with empty string (for random card)  # noqa: E501
                        mock_get_card_data.assert_called_once_with("")
