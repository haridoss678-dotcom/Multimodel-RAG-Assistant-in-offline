from rag_assistant.ingestion.language_detect import detect_language


def test_detect_english():
    assert detect_language("This is plain English text about decorators.") == "en"


def test_detect_tamil():
    assert detect_language("இது தமிழ் மொழியில் எழுதப்பட்ட உரை ஆகும்") == "ta"


def test_detect_mixed():
    text = "Decorators konjam explain pannunga தயவுசெய்து விளக்கவும் details ah"
    assert detect_language(text) == "mixed"


def test_detect_empty():
    assert detect_language("") == "en"
