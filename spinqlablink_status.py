"""Read-only status through SpinQ's official SpinQLabLink TCP API.

This file never opens a USB or COM port and never starts an experiment.
"""

import argparse
import getpass
import ipaddress
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="IP address shown by the SpinQ software")
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--account", required=True)
    parser.add_argument("--wait", type=int, default=5, help="Seconds to wait for status updates (1-30)")
    args = parser.parse_args()
    try:
        ipaddress.ip_address(args.host)
    except ValueError:
        parser.error("--host must be a numeric IP address")
    if not 1 <= args.port <= 65535 or not 1 <= args.wait <= 30:
        parser.error("--port must be 1-65535 and --wait must be 1-30")

    try:
        from spinqlablink import SpinQLabLink
    except ImportError:
        print("SpinQLabLink is missing. See README.md for optional setup.", file=sys.stderr)
        return 2

    password = getpass.getpass("SpinQ password (not saved): ")
    link = SpinQLabLink(args.host, args.port, args.account, password)
    seen = set()

    def on_update(device, update_type):
        seen.add(update_type)

    try:
        device = link.get_device()
        device.register_observer(on_update)
        link.connect()
        if not link.get_connection():
            print("TCP connection failed. Check the IP address and port.", file=sys.stderr)
            return 1
        if not link.wait_for_login(timeout=10):
            print("Login failed or timed out.", file=sys.stderr)
            return 1
        deadline = time.monotonic() + args.wait
        while time.monotonic() < deadline:
            time.sleep(0.2)
        print(json.dumps({
            "login": "ok",
            "status_received": "status" in seen,
            "lock_data_received": "lock_data" in seen,
            "device_status": link.get_device_status() if "status" in seen else None,
            "frequencies": link.get_device_frequencies() if "lock_data" in seen else None,
        }, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"Connection error: {exc}", file=sys.stderr)
        return 1
    finally:
        if link.get_connection():
            link.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
