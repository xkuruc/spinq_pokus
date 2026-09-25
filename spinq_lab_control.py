"""SpinQ Gemini Lab control through the official SpinQLabLink TCP API.

The tablet remains connected to the instrument by USB. This program talks to
the tablet's SpinQLabLink server over the local network; it never changes the
Windows USB driver or writes directly to the FTDI chip.
"""

import argparse
import csv
import getpass
import ipaddress
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


RABI_WIDTHS_US = (40, 80, 120, 160, 200)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=str)
    temporary.replace(path)


def check_tcp(host, port):
    try:
        with socket.create_connection((host, port), timeout=4):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"TCP {host}:{port} nie je dostupné ({exc}). "
            "Skontroluj adresu v aplikácii SpinQ a sieť Windows PC."
        ) from exc


def wait_for_status(link, seen, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and "status" not in seen:
        time.sleep(0.2)
    # The server may send frequencies shortly after the status message.
    if "status" in seen and "lock_data" not in seen:
        time.sleep(1)
    return {
        "status_received": "status" in seen,
        "lock_data_received": "lock_data" in seen,
        "device_status": link.get_device_status() if "status" in seen else None,
        "frequencies_hz": link.get_device_frequencies() if "lock_data" in seen else None,
    }


def require_ready(snapshot):
    if not snapshot["status_received"]:
        raise RuntimeError("Server neposlal stav prístroja; experiment sa nespustil.")
    status = snapshot["device_status"]
    if not status["connected"]:
        raise RuntimeError("Tablet hlási, že prístroj nie je pripojený; experiment sa nespustil.")
    if not status["lock_state"]:
        raise RuntimeError("Prístroj nehlási stabilný lock; experiment sa nespustil.")


def wait_for_experiment(link, experiment, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        # SpinQLabLink 1.0.2 exposes get_experiment_status(), but its
        # ExperimentManager has no method by that name. Read the registered
        # experiment's state, as the official examples do internally.
        if link.get_expMgr().current_experiment is not experiment:
            raise RuntimeError("SDK stratilo registráciu experimentu; skontroluj front na tablete.")
        state = experiment.get_status()
        if state in ("COMPLETED", "FAILED"):
            return experiment.get_result()
        if not link.get_connection():
            raise RuntimeError("Spojenie sa počas experimentu prerušilo.")
        time.sleep(1)
    raise TimeoutError(
        "Čakanie vypršalo. Úloha môže na prístroji stále bežať; "
        "pred ďalším spustením skontroluj jej stav na tablete."
    )


def run_one(link, experiment_type, configure, timeout):
    experiment, parameters = link.register_experiment(experiment_type)
    submitted = False
    finished = False
    try:
        configure(parameters)
        link.run_experiment()
        submitted = True
        result = wait_for_experiment(link, experiment, timeout)
        finished = True
        if result.get("state") != "COMPLETED":
            raise RuntimeError(f"Experiment skončil stavom {result.get('state')}: {result}")
        return result
    finally:
        # A timed-out or interrupted task may still be active on the tablet.
        # Keep its SDK registration until disconnect so late queue updates do
        # not access a missing experiment. Never retry it automatically.
        if not submitted or finished:
            link.deregister_experiment()


def install_queue_guard(link, queue_update_type):
    def handle_queue_update(data):
        # SDK 1.0.2 unconditionally dereferences current_experiment.id.
        # Queue updates can arrive just after deregistration/disconnect.
        current = link.get_expMgr().current_experiment
        if current is None:
            return
        for position, item in enumerate(data.get("queue", [])):
            if item.get("id") == current.id and position > 0:
                print(f"Experiment čaká vo fronte: {position} pred ním.", flush=True)

    link.handler_map[queue_update_type] = handle_queue_update


def run_rabi(link, types, report, report_path, timeout, pause):
    report["measurements"] = []
    if pause < 10:
        print(f"Prestávka medzi meraniami: {pause:g} s. "
              "Kratšia relaxácia môže skresliť Rabiho krivku.", flush=True)
    for width in RABI_WIDTHS_US:
        def configure(parameters):
            parameters.pulses = [types.Pulse(path=0, width=width, amplitude=100,
                                            phase=90, detuning=0)]
            parameters.makePps = True
            parameters.samplePath = 0
            parameters.custom_freq = False  # Use the device's lock frequency.

        print(f"Rabi: pulz {width} µs ...", flush=True)
        result = run_one(link, types.ExperimentType.RABI_OSCILLATIONS,
                         configure, timeout)
        real = result.get("result", {}).get("real")
        item = {"width_us": width, "real": real, "raw_result": result}
        report["measurements"].append(item)
        write_json(report_path, report)
        print(f"  výsledok: {real}", flush=True)
        if width != RABI_WIDTHS_US[-1]:
            time.sleep(pause)  # 10 s is the official Rabi example's pause.

    csv_path = report_path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("width_us", "real"))
        for item in report["measurements"]:
            writer.writerow((item["width_us"], item["real"]))
    print(f"Rabi CSV: {csv_path}")


def run_physical(link, types, report, report_path, timeout):
    def configure(parameters):
        parameters.type_setting = 0  # Documented customized measurement.
        parameters.pulses = [types.Pulse(path=0, width=40, amplitude=100,
                                        phase=90, detuning=0)]
        parameters.gradients = []
        parameters.relaxation_delay = 15  # Official physical-layer example.
        parameters.state_initialization = True
        parameters.sampleFre = 10000
        parameters.sampleCount = 16000
        parameters.sampleDelay = 0
        parameters.samplePath = 0
        parameters.h_freShift = 0
        parameters.p_freShift = 0
        parameters.h_freDemo = 0
        parameters.p_freDemo = 0

    print("Fyzikálna vrstva: jeden 40 µs pulz bez gradientu ...", flush=True)
    result = run_one(link, types.ExperimentType.PHYSICAL_LAYER_EXPERIMENT,
                     configure, timeout)
    report["physical_result"] = result
    write_json(report_path, report)
    graphs = result.get("result", {}).get("graph", [])
    print(f"  dokončené, počet blokov dát: {len(graphs)}", flush=True)
    if graphs:
        print("  dátové rady:", ", ".join(sorted(graphs[0])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("status", "rabi", "physical", "all"),
                        help="status iba číta; ostatné režimy spúšťajú merania")
    parser.add_argument("--host", required=True, help="IP z aplikácie SpinQ")
    parser.add_argument("--port", type=int, default=8181)
    parser.add_argument("--account", default="anyword",
                        help="predvolené prihlasovacie meno z príkladov výrobcu")
    parser.add_argument("--ask-password", action="store_true",
                        help="vypýtať heslo namiesto predvoleného 'anyword'")
    parser.add_argument("--status-wait", type=int, default=15)
    parser.add_argument("--timeout", type=int, default=300,
                        help="maximálne čakanie na jedno meranie v sekundách")
    parser.add_argument("--rabi-pause", type=float, default=10,
                        help="prestávka medzi Rabi meraniami v sekundách (štandardne 10)")
    args = parser.parse_args()
    try:
        ipaddress.ip_address(args.host)
    except ValueError:
        parser.error("--host musí byť IP adresa")
    if not 1 <= args.port <= 65535 or not 1 <= args.status_wait <= 60:
        parser.error("port musí byť 1–65535 a status-wait 1–60")
    if not 30 <= args.timeout <= 3600:
        parser.error("timeout musí byť 30–3600 sekúnd")
    if not 0 <= args.rabi_pause <= 60:
        parser.error("rabi-pause musí byť 0–60 sekúnd")

    try:
        from spinqlablink import ExperimentType, Pulse, SpinQLabLink
        from spinqlablink.utils.types import MachineType
    except ImportError as exc:
        print(f"Chýba SpinQLabLink: {exc}. Pozri README.md.", file=sys.stderr)
        return 2

    class Types:
        pass

    types = Types()
    types.ExperimentType = ExperimentType
    types.Pulse = Pulse
    password = getpass.getpass("Heslo SpinQ: ") if args.ask_password else "anyword"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    report_path = Path(__file__).resolve().parent / "results" / f"{stamp}_{args.mode}.json"
    report = {"mode": args.mode, "host": args.host, "port": args.port,
              "started_utc": stamp, "completed": False}
    link = None
    observer = None
    try:
        check_tcp(args.host, args.port)
        print(f"TCP {args.host}:{args.port} je dostupné.")
        link = SpinQLabLink(args.host, args.port, args.account, password)
        install_queue_guard(link, MachineType.MSG_POST_EXP_QUEUE_UPDATE)
        observed_updates = set()

        def observer(_device, update_type):
            observed_updates.add(update_type)

        link.get_device().register_observer(observer)
        link.connect()
        if not link.get_connection() or not link.wait_for_login(timeout=10):
            raise RuntimeError("Prihlásenie SpinQLabLink zlyhalo.")
        print("Prihlásenie SpinQLabLink: OK")
        report["device"] = wait_for_status(link, observed_updates, args.status_wait)
        write_json(report_path, report)
        print("Stav:", json.dumps(report["device"], ensure_ascii=False))

        if args.mode != "status":
            require_ready(report["device"])
        if args.mode in ("rabi", "all"):
            run_rabi(link, types, report, report_path, args.timeout, args.rabi_pause)
        if args.mode in ("physical", "all"):
            run_physical(link, types, report, report_path, args.timeout)

        report["completed"] = True
        write_json(report_path, report)
        print(f"Výsledky uložené: {report_path}")
        return 0
    except KeyboardInterrupt:
        report["error"] = "Prerušené používateľom; úloha môže na prístroji stále bežať."
        write_json(report_path, report)
        print(report["error"], file=sys.stderr)
        print(f"Doterajšie údaje: {report_path}", file=sys.stderr)
        return 130
    except Exception as exc:
        report["error"] = str(exc)
        write_json(report_path, report)
        print(f"Chyba: {exc}", file=sys.stderr)
        print(f"Doterajšie údaje: {report_path}", file=sys.stderr)
        return 1
    finally:
        if link is not None and observer is not None:
            link.get_device().unregister_observer(observer)
        if link is not None and link.get_connection():
            link.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
