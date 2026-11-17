"""Background GPU telemetry sampling through NVML.

Samples power, temperature, clocks, utilization, memory and clock-limiting
reasons while a benchmark runs. Power samples that are physically impossible
are dropped and counted: this laptop's driver sometimes reports ~590 W against a
60 W cap (see docs/progress.md, 2026-11-18).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .stats import summarize

# A reading above this multiple of the enforced power limit is treated as a glitch.
POWER_GLITCH_FACTOR = 1.5

# NVML clocks-event (throttle) reason bits that matter for benchmark validity.
_CLOCK_REASONS = {
    0x0000000000000001: "gpu_idle",
    0x0000000000000004: "sw_power_cap",
    0x0000000000000008: "hw_slowdown",
    0x0000000000000020: "sw_thermal",
    0x0000000000000040: "hw_thermal",
    0x0000000000000080: "hw_power_brake",
}


def is_valid_power(power_w: float, limit_w: Optional[float]) -> bool:
    """Reject negative readings and readings far above the enforced power limit."""
    if power_w < 0:
        return False
    if limit_w is None or limit_w <= 0:
        return True
    return power_w <= limit_w * POWER_GLITCH_FACTOR


def decode_clock_reasons(mask: int) -> list[str]:
    return sorted(name for bit, name in _CLOCK_REASONS.items() if mask & bit)


class GpuMonitor:
    """Samples NVML on a background thread. Safe to use when NVML is unavailable."""

    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self.available = False
        self.error: Optional[str] = None
        self._nvml = None
        self._handle = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.power_limit_w: Optional[float] = None
        self.mem_used_start_mib: Optional[float] = None
        self._power: list[float] = []
        self._power_rejected: list[float] = []
        self._temp: list[float] = []
        self._sm_clock: list[float] = []
        self._mem_clock: list[float] = []
        self._util: list[float] = []
        self._mem_used: list[float] = []
        self._pstates: set[int] = set()
        self._reasons_mask = 0

        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
            self.available = True
        except Exception as exc:  # NVML missing or driver not loaded
            self.error = f"{type(exc).__name__}: {exc}"

    def _clock_reasons(self) -> int:
        nv = self._nvml
        for name in ("nvmlDeviceGetCurrentClocksEventReasons", "nvmlDeviceGetCurrentClocksThrottleReasons"):
            fn = getattr(nv, name, None)
            if fn is not None:
                try:
                    return int(fn(self._handle))
                except Exception:
                    return 0
        return 0

    def _sample_once(self) -> None:
        nv, h = self._nvml, self._handle
        try:
            power_w = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            (self._power if is_valid_power(power_w, self.power_limit_w) else self._power_rejected).append(power_w)
        except Exception:
            pass
        try:
            self._temp.append(float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)))
            self._sm_clock.append(float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)))
            self._mem_clock.append(float(nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM)))
            self._util.append(float(nv.nvmlDeviceGetUtilizationRates(h).gpu))
            self._mem_used.append(nv.nvmlDeviceGetMemoryInfo(h).used / 2**20)
            self._pstates.add(int(nv.nvmlDeviceGetPerformanceState(h)))
        except Exception:
            pass
        self._reasons_mask |= self._clock_reasons()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def start(self) -> "GpuMonitor":
        if not self.available:
            return self
        self.mem_used_start_mib = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.available:
            self._sample_once()

    def summary(self) -> dict:
        if not self.available:
            return {"available": False, "error": self.error}

        def summ(values: list[float]) -> Optional[dict]:
            if not values:
                return None
            s = summarize(values)
            return {"mean": s["mean"], "min": s["min"], "max": s["max"]}

        mem_peak = max(self._mem_used) if self._mem_used else None
        return {
            "available": True,
            "interval_s": self.interval_s,
            "samples": len(self._temp),
            "power_limit_w": self.power_limit_w,
            "power_w": summ(self._power),
            "power_samples_rejected": len(self._power_rejected),
            "power_rejected_values_w": self._power_rejected[:10],
            "temperature_c": summ(self._temp),
            "sm_clock_mhz": summ(self._sm_clock),
            "mem_clock_mhz": summ(self._mem_clock),
            "utilization_pct": summ(self._util),
            "mem_used_start_mib": self.mem_used_start_mib,
            "mem_used_peak_mib": mem_peak,
            "mem_used_delta_peak_mib": (mem_peak - self.mem_used_start_mib)
            if mem_peak is not None and self.mem_used_start_mib is not None
            else None,
            "pstates_seen": sorted(f"P{p}" for p in self._pstates),
            "clock_limit_reasons": decode_clock_reasons(self._reasons_mask),
        }

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


def wait_for_cool_down(target_c: float = 50.0, timeout_s: float = 60.0, device_index: int = 0) -> Optional[float]:
    """Block until the GPU is at or below target_c, or timeout. Returns the final temperature."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:
        return None
    deadline = time.monotonic() + timeout_s
    temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    while temp > target_c and time.monotonic() < deadline:
        time.sleep(1.0)
        temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
    return temp
