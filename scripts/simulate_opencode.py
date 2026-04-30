
import json
import sys

import httpx


def simulate_opencode_request():
    print("--- [Simulator] Simulating OpenCode Anthropic Request ---")
    url = "http://localhost:8000/v1/messages"
    headers = {
        "x-api-key": "blend-test-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "blend",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Write a hello world in Python"}
        ],
        "stream": True
    }

    try:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=30.0) as response:
            if response.status_code != 200:
                print(f"FAILED: Status {response.status_code}")
                return False

            print("SUCCESS: Connection established, receiving stream...")
            has_content = False
            for line in response.iter_lines():
                print(f"DEBUG: {line}")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        event = json.loads(data_str)
                        if event.get("type") == "content_block_delta":
                            has_content = True
                            text = event.get("delta", {}).get("text", "")
                            print(text, end="", flush=True)
                    except:
                        continue
            print("\n--- [Simulator] Stream complete ---")
            return has_content
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    # Wait for server to be ready
    import time
    time.sleep(5)
    success = simulate_opencode_request()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
