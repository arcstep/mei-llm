"""Explicit survey views for related objects; not a task-gold compiler."""
from __future__ import annotations

from copy import deepcopy


def relation_units(value, kind: str):
    if kind == "crosswoz":
        if not isinstance(value, dict):
            raise ValueError("CrossWOZ requires a dialogue-ID mapping")
        for key, dialogue in value.items():
            if not isinstance(dialogue, dict) or not isinstance(dialogue.get("messages"), list):
                raise ValueError("CrossWOZ messages missing")
            visible, labels = [], []
            for message in dialogue["messages"]:
                if not isinstance(message.get("content"), str) or not isinstance(message.get("role"), str):
                    raise ValueError("CrossWOZ message role/content missing")
                visible.append({"role": message["role"], "content": message["content"]})
                labels.append({k: deepcopy(v) for k, v in message.items() if k not in {"role", "content"}})
            yield {"unit_id": str(key), "unit_kind": "whole-dialogue",
                   "input": {"messages": visible},
                   "annotations": {"messages": labels, "dialogue": {k: deepcopy(v) for k, v in dialogue.items() if k != "messages"}},
                   "original": deepcopy(dialogue)}
    elif kind == "json-schema-tests":
        if not isinstance(value, list):
            raise ValueError("schema test source must be an array")
        for index, group in enumerate(value):
            if "schema" not in group or not isinstance(group.get("tests"), list):
                raise ValueError("schema and test group required")
            cases, labels = [], []
            for test in group["tests"]:
                if "data" not in test or type(test.get("valid")) is not bool:
                    raise ValueError("test data and boolean expected result required")
                cases.append({k: deepcopy(v) for k, v in test.items() if k != "valid"})
                labels.append(test["valid"])
            yield {"unit_id": str(index), "unit_kind": "constraint-family",
                   "input": {"schema": deepcopy(group["schema"]), "description": group.get("description"), "cases": cases},
                   "annotations": {"expected_valid": labels, "label_origin": "upstream_test_not_reexecuted"},
                   "original": deepcopy(group)}
    elif kind == "table-bundle":
        if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
            raise ValueError("explicit table bundle with rows required")
        # Never reconstruct missing headers from semantic labels.
        visible = {k: deepcopy(value[k]) for k in ("rows", "headers", "schema", "constraints") if k in value}
        yield {"unit_id": str(value.get("id", "table")), "unit_kind": "table-relation-group",
               "input": visible, "annotations": deepcopy(value.get("annotations", {})), "original": deepcopy(value)}
    elif kind == "document-bundle":
        if not isinstance(value, dict) or not isinstance(value.get("relationships"), list):
            raise ValueError("explicit relationships required; shared directory is not a relationship")
        docs = value.get("documents")
        if not isinstance(docs, list) or any(not isinstance(d, dict) or "id" not in d for d in docs):
            raise ValueError("version-bound documents required")
        ids = {d["id"] for d in docs}
        if len(ids) != len(docs):
            raise ValueError("duplicate document IDs")
        for link in value["relationships"]:
            if link.get("from") not in ids or link.get("to") not in ids:
                raise ValueError("relationship endpoint missing")
        if any(not d.get("revision") or not d.get("sha256") for d in docs):
            raise ValueError("each document must bind revision and hash")
        yield {"unit_id": str(value.get("id", "bundle")), "unit_kind": "document-config-test",
               "input": {"documents": deepcopy(docs), "relationships": deepcopy(value["relationships"])},
               "annotations": deepcopy(value.get("annotations", {})), "original": deepcopy(value)}
    else:
        raise ValueError(f"unsupported relation kind: {kind}")
