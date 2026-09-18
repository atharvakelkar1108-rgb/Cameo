"""
Reads live system resource stats: CPU usage, available RAM, temperature
(where available), and battery (where available — laptops only, not the Pi
unless you've added a UPS HAT with a driver exposing it to psutil).

This is what feeds the Resource-Aware Model Selector (utils/model_selector.py)
and the web app's dashboard/device-monitoring panels.
"""

import psutil


def get_cpu_percent() -> float:
    """% CPU utilization averaged over a short window."""
    return psutil.cpu_percent(interval=0.3)


def get_cpu_core_count() -> int:
    return psutil.cpu_count(logical=True) or 1


def get_ram_stats_mb() -> dict:
    vm = psutil.virtual_memory()
    return {
        "total_mb": round(vm.total / (1024 * 1024), 1),
        "available_mb": round(vm.available / (1024 * 1024), 1),
        "used_mb": round(vm.used / (1024 * 1024), 1),
        "used_pct": vm.percent,
    }


def get_temperature_c():
    """
    Returns CPU temperature in Celsius if available, else None.
    Works out of the box on Raspberry Pi OS (via psutil's thermal_zone
    sensor) and on many Linux laptops. Not available on most Windows/macOS
    setups without extra drivers — None is a valid, expected result there.
    """
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        # Common sensor names across boards: 'cpu_thermal' (Pi), 'coretemp' (Intel laptops)
        for key in ("cpu_thermal", "coretemp", "cpu-thermal"):
            if key in temps and temps[key]:
                return round(temps[key][0].current, 1)
        # Fallback: just take the first sensor reported
        first_key = next(iter(temps))
        if temps[first_key]:
            return round(temps[first_key][0].current, 1)
    except (AttributeError, OSError):
        pass  # sensors_temperatures() doesn't exist on this OS (e.g. Windows)
    return None


def get_battery_stats():
    """
    Returns {percent, plugged_in} if a battery is detected (laptops, or a
    Pi with a UPS HAT exposing battery info to the OS), else None. The Pi
    Zero 2 W has no battery by default — None is the expected, correct
    result unless you've added power-monitoring hardware.
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return None
        return {"percent": battery.percent, "plugged_in": battery.power_plugged}
    except (AttributeError, OSError):
        return None


def get_system_stats() -> dict:
    """Single call used by the web app's dashboard — everything the
    Resource-Aware Model Selector and UI need in one shot."""
    ram = get_ram_stats_mb()
    return {
        "cpu_percent": get_cpu_percent(),
        "cpu_cores": get_cpu_core_count(),
        "ram_total_mb": ram["total_mb"],
        "ram_available_mb": ram["available_mb"],
        "ram_used_pct": ram["used_pct"],
        "temperature_c": get_temperature_c(),
        "battery": get_battery_stats(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(get_system_stats(), indent=2))
