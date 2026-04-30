import httpx
import asyncio
import json
import time

async def run_direct_task():
    url = "http://localhost:8000/v1/messages"
    headers = {
        "x-api-key": "blend-commercial-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Task: Write a complex class from scratch (no tools needed)
    task_description = """
    Create a production-ready Python implementation of a 'ResilienceManager'.
    It should include:
    1. An Exponential Backoff retry decorator with jitter.
    2. A Circuit Breaker with three states: CLOSED, OPEN, HALF_OPEN.
    3. Integration between the two: if the Circuit Breaker is OPEN, the retry should fail immediately.
    4. Type hints, docstrings, and a sample usage block.
    """
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": task_description}
        ],
        "stream": True
    }

    print(f"--- Direct Coding Task Simulation ---")
    start_time = time.time()
    full_content = ""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                print(f"Status: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data["type"] == "content_block_delta":
                            text = data["delta"].get("text", "")
                            full_content += text
                            print(".", end="", flush=True)
                
                print(f"\n--- Result Analysis ---")
                print(f"Total Time: {time.time() - start_time:.2f}s")
                print(f"Content Length: {len(full_content)} chars")
                
                if "class ResilienceManager" in full_content and "CircuitBreaker" in full_content:
                    print("VERIFICATION: SUCCESS. Complex logic generated and streamed.")
                else:
                    print("VERIFICATION: FAILED.")

    except Exception as e:
        print(f"\nConnection Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_direct_task())
