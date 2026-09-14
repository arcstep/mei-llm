"""Explicit survey views for related objects; not a task-gold compiler."""
from __future__ import annotations

from copy import deepcopy
import codecs
import json


def iter_json_object(stream, hasher, *, chunk_size=65536, record_limit=8 << 20, mapping=True):
    """Stream a top-level JSON mapping, preserving whole values and byte hash."""
    decoder = json.JSONDecoder()
    utf8 = codecs.getincrementaldecoder("utf-8")()
    buffer, position, ended = "", 0, False
    def fill():
        nonlocal buffer, position, ended
        if len(buffer) - position > record_limit:
            raise ValueError("individual JSON record exceeds streaming survey bound")
        raw = stream.read(chunk_size)
        hasher.update(raw)
        ended = not raw
        buffer = buffer[position:] + utf8.decode(raw, final=ended)
        position = 0
    def peek():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n": position += 1
            if position < len(buffer): return buffer[position]
            if ended: raise ValueError("unexpected end of JSON mapping")
            fill()
    def character(expected):
        nonlocal position
        if peek() != expected: raise ValueError(f"expected JSON delimiter {expected}")
        position += 1
    def value():
        nonlocal position
        peek()
        while True:
            try:
                item, stop = decoder.raw_decode(buffer, position)
                # A number can be a valid prefix at a buffer boundary.
                if not ended and (stop == len(buffer) or
                    (type(item) in (int, float) and buffer[stop:stop+1] in ("e", "E", "."))):
                    fill()
                    continue
                position = stop
                return item
            except json.JSONDecodeError:
                if ended: raise
                fill()
    character("{" if mapping else "[")
    keys = set()
    closing = "}" if mapping else "]"
    index = 0
    if peek() != closing:
        while True:
            key = value() if mapping else index
            if mapping:
                if not isinstance(key, str): raise ValueError("JSON mapping key is not a string")
                if key in keys: raise ValueError("duplicate JSON mapping key")
                keys.add(key)
                character(":")
            item = value()
            yield key, item
            index += 1
            if peek() == closing: break
            character(",")
    character(closing)
    while True:
        if buffer[position:].strip(): raise ValueError("trailing JSON content")
        position = len(buffer)
        if ended: return
        fill()


def relation_units(value, kind: str):
    if kind == 'wikisource-pages':
        pages=value.get('query',{}).get('pages',{})
        if not isinstance(pages,dict):raise ValueError('MediaWiki pages mapping required')
        for page in pages.values():
            if 'missing' in page:continue
            revision=page['revisions'][0]
            text=revision['slots']['main']['*']
            yield {'unit_id':str(page['pageid']),'unit_kind':'versioned-literary-page',
                   'input':{'title':page['title'],'wikitext':text},
                   'annotations':{'revision_id':revision['revid'],'timestamp':revision.get('timestamp'),
                                  'upstream_sha1':revision.get('sha1'),'categories':page.get('categories',[]),
                                  'transclusions_resolved':False,'complete_work_verified':False},'original':deepcopy(page)}
    elif kind == "risawoz":
        if not isinstance(value, list): raise ValueError("RiSAWOZ requires dialogue array")
        for row in value:
            turns = row.get("dialogue")
            if not isinstance(turns, list): raise ValueError("RiSAWOZ dialogue missing")
            visible, labels = [], []
            for t in turns:
                if any(not isinstance(t.get(k), str) for k in ('user_utterance','system_utterance')):
                    raise ValueError("RiSAWOZ utterance missing")
                visible.append({k:t[k] for k in ('user_utterance','system_utterance')})
                labels.append({k:deepcopy(v) for k,v in t.items() if k not in visible[-1]})
            yield {"unit_id":str(row['dialogue_id']),"unit_kind":"whole-dialogue",
                   "input":{"turns":visible},"annotations":{"turns":labels,"dialogue":{k:deepcopy(v) for k,v in row.items() if k!='dialogue'}},"original":deepcopy(row)}
    elif kind == "naturalconv":
        if not isinstance(value, list): raise ValueError("NaturalConv requires dialogue array")
        for row in value:
            if not isinstance(row.get('content'), list) or any(not isinstance(t,str) for t in row['content']):
                raise ValueError("NaturalConv content missing")
            yield {"unit_id":str(row['dialog_id']),"unit_kind":"whole-dialogue",
                   "input":{"turns":deepcopy(row['content'])},
                   "annotations":{"document_id":row.get('document_id'),"grounding_document_loaded":False},"original":deepcopy(row)}
    elif kind == "toolace":
        if not isinstance(value, list): raise ValueError("ToolACE requires a record array")
        for index, row in enumerate(value):
            turns = row.get("conversations")
            if not isinstance(turns, list) or not isinstance(row.get("system"), str):
                raise ValueError("ToolACE system and conversations required")
            if any(not isinstance(t, dict) or not isinstance(t.get("from"), str) or
                   not isinstance(t.get("value"), str) for t in turns):
                raise ValueError("invalid ToolACE turn")
            yield {"unit_id": str(index), "unit_kind": "whole-tool-dialogue",
                   "input": {"system_and_tool_descriptions": row["system"],
                             "user_turns": [{"turn_index": i, **deepcopy(t)} for i,t in enumerate(turns) if t["from"] == "user"]},
                   "annotations": {"non_user_turns": [{"turn_index": i, **deepcopy(t)} for i,t in enumerate(turns) if t["from"] != "user"],
                                   "label_origin": "upstream_generated_not_reexecuted",
                                   "view_scope": "survey only; user turns are not one simultaneous agent input"},
                   "original": deepcopy(row)}
    elif kind == "schema-document":
        if not isinstance(value, (dict, bool)):
            raise ValueError("JSON schema must be an object or boolean")
        yield {"unit_id": str(value.get("$id", "schema")) if isinstance(value, dict) else "boolean-schema",
               "unit_kind": "schema-document", "input": {"schema": deepcopy(value)},
               "annotations": {}, "original": deepcopy(value)}
    elif kind == "crosswoz":
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
