from buc_factory.utils import fmt


def test_basic_substitution():
    assert fmt("Hello {name}!", name="world") == "Hello world!"


def test_multiple_keys():
    assert fmt("{a} + {b} = {c}", a=1, b=2, c=3) == "1 + 2 = 3"


def test_no_placeholders():
    assert fmt("no placeholders") == "no placeholders"


def test_unused_kwargs_ignored():
    assert fmt("Hello {name}", name="Alice", extra="ignored") == "Hello Alice"
