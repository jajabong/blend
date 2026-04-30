import httpx
import asyncio
import json
import time

async def run_complex_task():
    url = "http://localhost:8000/v1/messages"
    headers = {
        "x-api-key": "blend-commercial-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Complex task: Refactor budget and add tests
    task_description = """
    I need you to perform a refactoring task on the 'blend' codebase:
    1. Locate 'blend/core/budget.py'.
    2. Add a method 'get_global_usage_report()' to the 'ResourceModel' class. This method should return a dictionary containing the total 'completion_tokens' and 'prompt_tokens' across ALL models tracked in the persistent storage.
    3. Ensure the code follows the existing typing and style.
    4. Provide the code for a new test file 'tests/test_budget_report.py' that would verify this new method works correctly.
    """
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": task_description}
        ],
        "stream": True
    }

    print(f"--- Launching Complex Engineering Task Simulation ---")
    print(f"Task: Refactoring & Test Generation")
    
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
                            # Optional: print dots to show progress
                            print(".", end="", flush=True)
                        elif data["type"] == "message_stop":
                            print("\n[Stream Complete]")

                print(f"\n--- Result Analysis ---")
                print(f"Total Time: {time.time() - start_time:.2f}s")
                
                # Check for key architectural markers in output
                has_method = "get_global_usage_report" in full_content
                has_test = "tests/test_budget_report.py" in full_content or "import pytest" in full_content
                
                if has_method and has_test:
                    print("VERIFICATION: SUCCESS. The model correctly planned, refactored, and generated tests.")
                else:
                    print(f"VERIFICATION: PARTIAL. Method found: {has_method}, Test found: {has_test}")
                
                print("\n--- Content Snippet ---")
                print(full_content[:500] + "...")

    except Exception as e:
        print(f"\nConnection Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_complex_task())
