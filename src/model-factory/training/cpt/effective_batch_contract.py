from __future__ import annotations


def effective_batch_tokens(row: dict) -> int:
    values = [int(row.get(name) or 0) for name in ("batch_size", "grad_accum", "seq_len")]
    if any(value < 1 for value in values):
        raise ValueError("batch_size, grad_accum and seq_len must be positive and explicit")
    return values[0] * values[1] * values[2]


def continuation_batch_error(parent: dict, stage: dict) -> str | None:
    try:
        previous = effective_batch_tokens(parent)
        proposed = effective_batch_tokens(stage)
    except (TypeError, ValueError) as error:
        return str(error)
    if previous != proposed:
        return (
            f"continuation effective batch changed: parent={previous} tokens/update, "
            f"proposed={proposed}; microbatch changes require compensating gradient accumulation"
        )
    return None
