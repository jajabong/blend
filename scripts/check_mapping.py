def _map_finish_reason(finish_reason):
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    return mapping.get(finish_reason, "end_turn")

print(_map_finish_reason("stop"))
print(_map_finish_reason("length"))
print(_map_finish_reason("none"))
