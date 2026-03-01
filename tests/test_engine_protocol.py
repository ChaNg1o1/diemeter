"""Tests for engine JSON-RPC protocol helpers."""

from __future__ import annotations

import pytest

from diemeter.engine.protocol import RPCError, parse_request


def test_parse_request_success() -> None:
    rid, method, params = parse_request(
        {"id": "1", "method": "engine.ping", "params": {"foo": "bar"}}
    )
    assert rid == "1"
    assert method == "engine.ping"
    assert params == {"foo": "bar"}


def test_parse_request_without_id() -> None:
    rid, method, params = parse_request({"method": "snapshot.get"})
    assert rid is None
    assert method == "snapshot.get"
    assert params == {}


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {"method": "", "params": {}},
        {"method": 1, "params": {}},
        {"method": "engine.ping", "params": []},
    ],
)
def test_parse_request_invalid(payload) -> None:
    with pytest.raises(RPCError):
        parse_request(payload)
