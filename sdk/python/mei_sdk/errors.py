from __future__ import annotations

from .version import load_json

_CODES = {row["id"]: row for row in load_json("errors.json")["codes"]}


class SdkError(Exception):
    def __init__(self, code_id: str, message: str | None = None):
        row = _CODES[code_id]
        self.code = int(row["code"])
        self.id = code_id
        self.message = message or str(row["message"])
        super().__init__(self.message)

    def as_dict(self) -> dict:
        return {"code": self.code, "id": self.id, "message": self.message}


def error_from_id(code_id: str, message: str | None = None) -> dict:
    return SdkError(code_id, message).as_dict()


def lookup_id(code: int) -> str:
    for row in _CODES.values():
        if int(row["code"]) == int(code):
            return str(row["id"])
    return "invalid_argument"
