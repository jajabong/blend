
import json
import os
import time


def monitor_health(health_file=".blend_health.json", interval=5):
    print(f"--- [Watchdog] Monitoring started on {health_file} ---")
    try:
        while True:
            if os.path.exists(health_file):
                with open(health_file) as f:
                    data = json.load(f)
                    providers = data.get("providers", {})
                    for p, status in providers.items():
                        if status.get("state") == "OPEN":
                            print(f"ALERT: Provider {p} is FAILED/OPEN. Switching to backup!")
                        else:
                            print(f"Status: {p} is HEALTHY")
            else:
                print("--- [Watchdog] Health file not found, waiting... ---")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("--- [Watchdog] Stopped ---")

if __name__ == "__main__":
    monitor_health()
