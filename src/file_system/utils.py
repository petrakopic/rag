#  Collection of clean-up functions for file processing which should be done on file before
#  ingesting them into the Qdrant vector database.

import re

# TODO: recognize and remove headers and footers automatically
HEADER_1 = "SCCS /1644/22 Final version CORRIGEND UM 21 March 20 23"
HEADER_2 = "Opinion on the safety of aluminiu m in cosmetic p roducts - Submission III"


def remove_header(text) -> str:
    """
    Removes all occurrences of the specified header from the text.

    Parameters:
    - text (str): The text from which the header is to be removed.
    - header (str): The header string to be removed.

    Returns:
    - str: The text with the headers removed.
    """

    text = re.sub(r"\s+", " ", text)
    for header in [HEADER_1, HEADER_2]:
        text = re.sub(re.escape(header), "", text)
    return text
