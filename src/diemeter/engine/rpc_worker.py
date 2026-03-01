"""stdio JSON-RPC worker for the headless diemeter engine."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, Optional

from .controller import EngineController
from .protocol import RPCError, error_response, parse_request, success_response


def _write_message(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _map_exception(exc: Exception) -> RPCError:
    if isinstance(exc, RPCError):
        return exc
    if isinstance(exc, FileNotFoundError):
        return RPCError("not_found", str(exc))
    if isinstance(exc, ValueError):
        return RPCError("invalid_params", str(exc))
    return RPCError(
        "internal_error",
        str(exc) or exc.__class__.__name__,
        data={"traceback": traceback.format_exc(limit=8)},
    )


def run_worker() -> int:
    controller = EngineController()

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        request_id: Optional[str] = None

        try:
            payload = json.loads(line)
            request_id, method, params = parse_request(payload)
            result = controller.handle(method, params)
            _write_message(success_response(request_id, result))
        except Exception as exc:
            err = _map_exception(exc)
            _write_message(error_response(request_id, err))

    return 0


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
