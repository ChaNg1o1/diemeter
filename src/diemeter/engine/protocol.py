"""JSON-RPC protocol helpers for diemeter engine worker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class RPCError(Exception):
    """Structured RPC error."""

    code: str
    message: str
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.data:
            payload["data"] = self.data
        return payload


def parse_request(message: Any) -> Tuple[Optional[str], str, Dict[str, Any]]:
    if not isinstance(message, dict):
        raise RPCError("invalid_request", "Request must be a JSON object")

    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params", {})

    if not isinstance(method, str) or not method:
        raise RPCError("invalid_request", "'method' must be a non-empty string")

    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise RPCError("invalid_params", "'params' must be an object")

    if request_id is not None:
        request_id = str(request_id)

    return request_id, method, params


def success_response(request_id: Optional[str], result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": request_id,
        "result": result,
    }


def error_response(request_id: Optional[str], error: RPCError) -> Dict[str, Any]:
    return {
        "id": request_id,
        "error": error.to_dict(),
    }
