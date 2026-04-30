import httpx
import asyncio
import time

async def simulate_opencode():
    url = "http://localhost:8000/v1/messages"
    headers = {
        "x-api-key": "blend-commercial-token",
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    payload = {
        "model": "claude-3-opus-20240229",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Tell me a short story."}],
        "stream": True
    }
    
    print(f"Sending request to {url}...")
    start_time = time.time()
    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"Status: {response.status_code}")
            async for line in response.aiter_lines():
                if line:
                    print(line)
                    if "message_stop" in line:
                        break
    print(f"Time taken: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    asyncio.run(simulate_opencode())
