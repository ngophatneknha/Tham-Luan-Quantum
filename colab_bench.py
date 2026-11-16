"""Benchmark mo phong mach luong tu (Colab/Linux/Windows). Moi lan do = 1 tien trinh rieng.
Chay: python colab_bench.py            -> do day du, ghi bench_colab.json
      python colab_bench.py worker SIM N  (noi bo)"""
import sys, json, time, subprocess, os
import numpy as np

D = 10  # so lop mach


def angles(n):
    return np.random.default_rng(123).uniform(0, 2 * np.pi, (D, n))


def sim_array(xp, dtype, n):
    th = angles(n)
    psi = xp.zeros(2 ** n, dtype=dtype); psi[0] = 1
    for l in range(D):
        for k in range(n):
            c, s = np.cos(th[l, k] / 2), np.sin(th[l, k] / 2)
            v = psi.reshape(2 ** k, 2, -1)
            a = v[:, 0, :].copy(); b = v[:, 1, :].copy()
            v[:, 0, :] = c * a - s * b
            v[:, 1, :] = s * a + c * b
        for c_ in range(n - 1):
            t_ = c_ + 1
            v = psi.reshape(2 ** c_, 2, 1, 2, 2 ** (n - t_ - 1))
            tmp = v[:, 1, :, 0, :].copy()
            v[:, 1, :, 0, :] = v[:, 1, :, 1, :]
            v[:, 1, :, 1, :] = tmp
    pr = psi.real ** 2 + psi.imag ** 2
    h = 2 ** (n - 1)
    return float(pr[:h].sum() - pr[h:].sum())


def make(sim, n):
    th = angles(n)
    if sim == "numpy":
        return lambda: sim_array(np, np.complex128, n)
    if sim in ("cupy64", "cupy128"):
        import cupy as cp
        dt = cp.complex64 if sim == "cupy64" else cp.complex128

        def f():
            r = sim_array(cp, dt, n); cp.cuda.Stream.null.synchronize(); return r
        return f
    if sim.startswith("pl_"):
        import pennylane as qml
        name = {"pl_default": "default.qubit", "pl_lightning": "lightning.qubit", "pl_gpu": "lightning.gpu"}[sim]
        dev = qml.device(name, wires=n)

        @qml.qnode(dev)
        def circ():
            for l in range(D):
                for k in range(n): qml.RY(th[l, k], wires=k)
                for c in range(n - 1): qml.CNOT(wires=[c, c + 1])
            return qml.expval(qml.PauliZ(0))
        return lambda: float(circ())
    if sim in ("aer", "aer_gpu"):
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(n)
        for l in range(D):
            for k in range(n): qc.ry(float(th[l, k]), k)
            for c in range(n - 1): qc.cx(c, c + 1)
        qc.save_expectation_value(SparsePauliOp("I" * (n - 1) + "Z"), list(range(n)))
        be = AerSimulator(method="statevector", device="GPU" if sim == "aer_gpu" else "CPU")
        return lambda: float(be.run(qc).result().data(0)["expectation_value"])
    raise ValueError(sim)


def peak_ram_mb():
    import psutil
    mi = psutil.Process().memory_info()
    if hasattr(mi, "peak_wset"):  # Windows
        return mi.peak_wset / 2 ** 20
    try:  # Linux: VmHWM cua chinh tien trinh nay (ru_maxrss bi ke thua tu tien trinh cha sau fork)
        for l in open("/proc/self/status"):
            if l.startswith("VmHWM"): return int(l.split()[1]) / 1024
    except Exception:
        pass
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def worker(sim, n):
    make(sim, 4)()  # khoi dong
    f = make(sim, n)
    reps = 3 if n <= 18 else 1
    ts, val = [], None
    for _ in range(reps):
        t0 = time.perf_counter(); val = f(); ts.append(time.perf_counter() - t0)
    print(json.dumps({"sim": sim, "n": n, "sec": float(np.median(ts)), "val": val, "ram_mb": peak_ram_mb()}))


def usable(sim):
    """Thu nhanh xem bo mo phong co chay duoc khong (GPU co the khong co)."""
    try:
        make(sim, 4)(); return True
    except Exception as e:
        print("bo qua", sim, "->", str(e)[:100]); return False


FACTOR = {"numpy": 4.0, "pl_default": 3.5, "pl_lightning": 1.4, "aer": 1.4}  # he so bo nho ~ so lan state vector


def hardware():
    import platform, psutil
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
            "total_ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
            "avail_ram_gb_at_start": round(psutil.virtual_memory().available / 2 ** 30, 1)}
    try:
        info["cpu"] = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        info["cpu"] = platform.processor()
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
    except Exception:
        info["gpu"] = ""
    import importlib.metadata as md
    for p in ("qiskit", "qiskit-aer", "qiskit-aer-gpu-cu12", "pennylane", "pennylane-lightning", "pennylane-lightning-gpu", "numpy", "cupy-cuda12x"):
        try: info[p] = md.version(p)
        except Exception: pass
    return info


OUT = os.environ.get("OUT", "bench_colab.json")


def driver():
    import psutil
    info = hardware()
    print(json.dumps(info, indent=1), flush=True)
    out = {"info": info, "runs": []}
    want = os.environ.get("SIMS", "numpy,pl_default,pl_lightning,aer,aer_gpu,pl_gpu,cupy64").split(",")
    sims = [s for s in want if (s in ("aer_gpu", "pl_gpu", "cupy64", "cupy128") and not info["gpu"]) is False and usable(s)]
    nmax = int(os.environ.get("NMAX", 30))
    ns = [n for n in (10, 12, 14, 16, 18, 20, 22, 24, 25, 26, 27, 28, 29, 30) if n <= nmax]
    vram_mb = 0
    if info["gpu"]:
        try: vram_mb = float(info["gpu"].split(",")[1].replace("MiB", "").strip())
        except Exception: pass
    for sim in sims:
        for n in ns:
            avail = psutil.virtual_memory().available / 2 ** 20
            state = 16 * 2 ** n / 2 ** 20
            if sim in FACTOR and state * FACTOR[sim] > avail * 0.85:
                print("skip(ram)", sim, n, flush=True); break
            if sim in ("aer_gpu", "pl_gpu") and state * 1.3 > vram_mb * 0.9:
                print("skip(vram)", sim, n, flush=True); break
            if sim in ("cupy64", "cupy128") and state / (2 if sim == "cupy64" else 1) * 2.5 > vram_mb * 0.9:
                print("skip(vram)", sim, n, flush=True); break
            try:
                r = subprocess.run([sys.executable, __file__, "worker", sim, str(n)], capture_output=True, text=True, timeout=240)
            except subprocess.TimeoutExpired:
                print("timeout", sim, n, flush=True); out["runs"].append({"sim": sim, "n": n, "status": "timeout"}); break
            if r.returncode != 0 or not r.stdout.strip():
                print("fail", sim, n, r.stderr[-200:].replace("\n", " "), flush=True)
                out["runs"].append({"sim": sim, "n": n, "status": "fail"}); break
            rec = json.loads(r.stdout.strip().splitlines()[-1]); rec["status"] = "ok"; out["runs"].append(rec)
            print(sim, n, "%.3fs" % rec["sec"], "val=%.6f" % rec["val"], "ram=%.0fMB" % rec["ram_mb"], flush=True)
            json.dump(out, open(OUT, "w"), indent=1)
            if rec["sec"] > 90: break
    json.dump(out, open(OUT, "w"), indent=1)
    print("XONG. File ket qua:", OUT)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        driver()
