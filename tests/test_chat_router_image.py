from services.chat_router import extract_image_request


def test_detects_show_me_memes():
    assert extract_image_request("show me cat memes") == "cat memes"

def test_detects_send_me():
    assert extract_image_request("send me a cooking guide") == "cooking guide"

def test_detects_got_any():
    assert extract_image_request("got any dog gifs") == "dog gifs"

def test_detects_find_me():
    assert extract_image_request("find me funny cat pics") == "funny cat pics"

def test_detects_post():
    assert extract_image_request("post some memes") == "memes"

def test_no_match_returns_none():
    assert extract_image_request("how ya feeling") is None

def test_no_match_plain_chat():
    assert extract_image_request("what time is it") is None

def test_detects_share():
    assert extract_image_request("share a cat meme with me") == "cat meme"

def test_detects_let_me_see():
    assert extract_image_request("let me see a funny cat meme") == "funny cat meme"

def test_detects_can_you_show():
    assert extract_image_request("can you show me dog memes") == "dog memes"

def test_detects_wanna_see():
    assert extract_image_request("wanna show me some cat gifs") == "cat gifs"

def test_detects_let_me_another():
    result = extract_image_request("let me another funny cat meme you like")
    assert result is not None
    assert "cat" in result
    assert "meme" in result

def test_no_match_how_to():
    # "show me how to cook" should NOT trigger — no media keyword
    assert extract_image_request("show me how to cook") is None
