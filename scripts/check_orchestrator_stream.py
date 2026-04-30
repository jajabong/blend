import asyncio
from blend.core.orchestrator import BlendOrchestrator

async def main():
    orch = BlendOrchestrator()
    msgs = [{"role": "user", "content": "Hello"}]
    print("Testing orchestrator.stream_messages directly...")
    for chunk in orch.stream_messages(msgs):
        print(f"Chunk: {chunk}")

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.getcwd())
    asyncio.run(main())
