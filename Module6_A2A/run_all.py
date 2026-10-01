"""Start the registry, both agents and the supervisor in one terminal.

Each one is an ordinary uvicorn server, so you can equally well run the four
files in four terminals if you would rather watch their logs separately.
Press Ctrl+C to stop them all.
"""

import socket
import subprocess
import sys
import time

from config import (
    CURRENCY_AGENT_PORT,
    HOST,
    REGISTRY_PORT,
    SUPERVISOR_PORT,
    TIME_AGENT_PORT,
    WEATHER_AGENT_PORT,
)

SERVICES = [
    ("registry", "registry.py", REGISTRY_PORT),
    ("weather agent (LangChain)", "agent_weather_langchain.py", WEATHER_AGENT_PORT),
    ("currency agent (Google ADK)", "agent_currency_adk.py", CURRENCY_AGENT_PORT),
    ("time agent (plain Python)", "agent_time_plain.py", TIME_AGENT_PORT),
    ("supervisor (LangGraph)", "supervisor.py", SUPERVISOR_PORT),
]


def wait_until_listening(port: int, timeout: float = 120.0) -> bool:
    """Poll a port until something answers. Importing ADK can take a while."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex((HOST, port)) == 0:
                return True
        time.sleep(0.5)
    return False


processes = []
try:
    for label, script, port in SERVICES:
        processes.append(subprocess.Popen([sys.executable, script]))
        print(f"starting {label} on http://{HOST}:{port}")

    print()
    for label, _, port in SERVICES:
        state = "ready" if wait_until_listening(port) else "FAILED to start"
        print(f"{label}: {state}")

    print("\nIn another terminal now run:  python demo.py")
    for process in processes:
        process.wait()
except KeyboardInterrupt:
    print("\nshutting down")
finally:
    for process in processes:
        process.terminate()
