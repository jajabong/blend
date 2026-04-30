from blend.core.orchestrator import BlendOrchestrator
orchestrator = BlendOrchestrator()
messages = [{"role": "user", "content": "Hello"}]
for chunk in orchestrator.stream_messages(messages=messages):
    print(f"Chunk: {chunk}")
    if "finish_reason" in chunk:
        print(f"Finish reason: {chunk['finish_reason']}")
