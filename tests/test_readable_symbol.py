import pytest

from app.logic import readable_symbol, readable_text


@pytest.mark.parametrize(
    "raw, words",
    [
        ("NIFTY26O1923000CE", "NIFTY 23,000 CE · 19 Oct"),
        ("NIFTY26OCT23000CE", "NIFTY 23,000 CE · Oct monthly"),
        ("SENSEX26N0580000PE", "SENSEX 80,000 PE · 5 Nov"),
        ("SENSEX26DEC80000PE", "SENSEX 80,000 PE · Dec monthly"),
        ("BANKNIFTY25JAN51100CE", "BANKNIFTY 51,100 CE · Jan monthly"),
        ("NIFTY2510223800CE", "NIFTY 23,800 CE · 2 Jan"),
        ("NIFTY26D1022000PE", "NIFTY 22,000 PE · 10 Dec"),
    ],
)
def test_option_symbols_read_in_words(raw, words):
    assert readable_symbol(raw) == words


def test_joined_legs_convert_each_part():
    assert readable_symbol("NIFTY26O1923000CE + NIFTY26O1923200PE") == (
        "NIFTY 23,000 CE · 19 Oct + NIFTY 23,200 PE · 19 Oct"
    )


@pytest.mark.parametrize("raw", ["NIFTY FUT", "NIFTY", "RELIANCE", "", "GOLD26DECFUT"])
def test_anything_else_is_left_alone(raw):
    assert readable_symbol(raw) == raw


def test_symbols_inside_a_sentence_are_converted():
    text = "Bought 1 lot of NIFTY26O1923000CE at Rs 10.00 -- it expired."
    assert readable_text(text) == (
        "Bought 1 lot of NIFTY 23,000 CE · 19 Oct at Rs 10.00 -- it expired."
    )
    assert readable_text("Bought 5 shares of RELIANCE") == "Bought 5 shares of RELIANCE"
