from blend.core.orchestrator import BlendOrchestrator
orchestrator = BlendOrchestrator()
messages = [{"role": "user", "content": "Hello"}]
for chunk in orchestrator.stream_messages(messages=messages):
    print(chunk)
