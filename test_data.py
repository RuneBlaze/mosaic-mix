"""Unit tests for data.py"""
from data import DictDotAccess


def test_dict_dot_access():
    d = DictDotAccess({"a": 1, "b": 2})
    assert d.a == 1
    assert d.b == 2
    assert d.a == d["a"]
    assert not hasattr(d, "c")
    assert hasattr(d, "a")
