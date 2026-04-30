import httpx
import asyncio
import json
import time

async def test_integration():
    url = "http://localhost:8000/v1/messages"
    headers = {
        "x-api-key": "blend-commercial-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Write a quick python function to calculate fibonacci."}
        ],
        "stream": True
    }

    print(f"--- Starting Integration Test ---")
    print(f"Target URL: {url}")
    print(f"Using Model: {payload['model']}")
    
    start_time = time.time()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                print(f"Status Code: {response.status_code}")
                if response.status_code != 200:
                    print(f"Error: {await response.aread()}")
                    return

                event_count = 0
                full_content = ""
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        event_count += 1
                        data_str = line[6:]
                        try:
                            event = json.loads(data_str)
                            event_type = event.get("type")
                            
                            if event_type == "content_block_delta":
                                delta = event.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    full_content += delta.get("text", "")
                            
                            # Print first few events and the final stop event for visibility
                            if event_count <= 3 or event_type == "message_stop":
                                print(f"Event {event_count} [{event_type}]: {data_str[:80]}...")
                                
                        except json.JSONDecodeError:
                            print(f"Failed to decode SSE line: {line}")

                print(f"--- Test Finished ---")
                print(f"Total Events Received: {event_count}")
                print(f"Total Time: {time.time() - start_time:.2f}s")
                print(f"Sample Content Output:\n{full_content[:200]}...")
                
                if event_count > 5 and len(full_content) > 20:
                    print("\nRESULT: SUCCESS - Integration protocol is healthy.")
                else:
                    print("\nRESULT: FAILED - Stream was too short or empty.")

    except Exception as e:
        print(f"Connection Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_integration())
