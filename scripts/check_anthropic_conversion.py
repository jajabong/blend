import json

# Mimic a chunk from orchestrator.stream_messages
chunk = {
    "id": "chatcmpl-12345",
    "choices": [{
        "delta": {"content": "Hello world"},
        "finish_reason": "stop"
    }]
}

# Mimic _convert_chunk_to_anthropic_events
def _convert_chunk_to_anthropic_events(chunk, is_first=True):
    chunk_id = chunk.get("id", "msg_anthropic")
    choices = chunk.get("choices", [])
    choice = choices[0] if choices else {}
    delta = choice.get("delta", {})
    finish_reason = choice.get("finish_reason")

    if is_first:
        yield json.dumps({
            "type": "message_start",
            "message": {
                "id": chunk_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "blend",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })
        yield json.dumps({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })

    if "content" in delta and delta["content"]:
        yield json.dumps({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": delta["content"]},
        })

    if finish_reason and finish_reason != "null":
        yield json.dumps({"type": "content_block_stop", "index": 0})
        yield json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": finish_reason, "stop_sequence": None},
            "usage": {"output_tokens": 0},
        })
        yield json.dumps({"type": "message_stop"})

# Run test
for event in _convert_chunk_to_anthropic_events(chunk, True):
    print(f"data: {event}\n\n")

