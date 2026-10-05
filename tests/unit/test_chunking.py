from rag_assistant.ingestion.chunking import CHUNK_SIZE, chunk_elements
from rag_assistant.ingestion.models import ContentElement


def _page(text: str) -> ContentElement:
    return ContentElement(element_type="paragraph", text=text, page=1)


_TABLE = "| Item | Cost |\n|------|------|\n| DC   | 100  |\n| GRN  | 250  |\n"


def test_markdown_table_is_one_chunk_tagged_table():
    """CLAUDE.md §7 item 33 - nothing ever set element_type="table", so
    §20.1's table_lookup/fact_lookup routing filtered an always-empty set."""
    chunks = chunk_elements([_page(f"Intro sentence.\n\n{_TABLE}\nClosing sentence.")], "src")

    tables = [c for c in chunks if c.element_type == "table"]
    assert len(tables) == 1
    # Header row and every body row stay together - half a table is useless.
    assert "| Item | Cost |" in tables[0].text
    assert "| GRN  | 250  |" in tables[0].text


def test_a_long_table_is_still_not_split():
    rows = "".join(f"| row{i} | {i} |\n" for i in range(400))
    big_table = f"| Item | Cost |\n|------|------|\n{rows}"
    assert len(big_table) > CHUNK_SIZE

    chunks = chunk_elements([_page(big_table)], "src")

    tables = [c for c in chunks if c.element_type == "table"]
    assert len(tables) == 1
    assert tables[0].text.count("| row") == 400


def test_prose_containing_a_pipe_is_not_mistaken_for_a_table():
    """The separator row is required, so ordinary text with a pipe character
    stays prose rather than being swallowed as a table."""
    chunks = chunk_elements([_page("Use the | operator to pipe output between commands.")], "src")

    assert all(c.element_type != "table" for c in chunks)


def test_chunk_index_follows_document_order():
    """Parent reassembly sorts on chunk_index; without it PDF chunks were
    ordered by uuid4 (CLAUDE.md §7 item 27)."""
    long_prose = "sentence. " * 400
    chunks = chunk_elements([_page(long_prose)], "src")

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_short_element_produces_one_chunk():
    elements = [ContentElement(element_type="paragraph", text="short text", page=1)]
    chunks = chunk_elements(elements, source_id="abc")
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].source_id == "abc"


def test_chunk_long_element_splits_with_overlap():
    long_text = "word " * 500  # well over CHUNK_SIZE characters
    elements = [ContentElement(element_type="paragraph", text=long_text, page=3)]
    chunks = chunk_elements(elements, source_id="abc")
    assert len(chunks) > 1
    assert all(c.page == 3 for c in chunks)


def test_fenced_code_block_is_never_split_and_is_tagged_as_code():
    """CLAUDE.md §3: a code-lookup question must return the code block intact.
    The block here is deliberately far longer than CHUNK_SIZE, which a plain
    fixed-size split would cut in half."""
    code = "```python\n" + "print('hello world')\n" * 200 + "```"
    text = f"Here is an example.\n\n{code}\n\nThat is the example."
    elements = [ContentElement(element_type="paragraph", text=text, page=1)]

    chunks = chunk_elements(elements, source_id="abc")
    code_chunks = [c for c in chunks if c.element_type == "code"]

    assert len(code_chunks) == 1
    assert code_chunks[0].text.startswith("```python")
    assert code_chunks[0].text.endswith("```")
    assert code_chunks[0].text.count("print('hello world')") == 200
    assert len(code_chunks[0].text) > 1000  # intact despite exceeding CHUNK_SIZE


def test_prose_around_a_code_block_is_still_chunked_normally():
    code = "```\nx = 1\n```"
    text = f"{'prose ' * 400}\n\n{code}\n\n{'more prose ' * 400}"
    elements = [ContentElement(element_type="paragraph", text=text, page=1)]

    chunks = chunk_elements(elements, source_id="abc")

    assert sum(1 for c in chunks if c.element_type == "code") == 1
    assert sum(1 for c in chunks if c.element_type == "paragraph") > 1


def test_chunk_never_spans_two_pages():
    elements = [
        ContentElement(element_type="paragraph", text="page one content", page=1),
        ContentElement(element_type="paragraph", text="page two content", page=2),
    ]
    chunks = chunk_elements(elements, source_id="abc")
    pages = {c.page for c in chunks}
    assert pages == {1, 2}
