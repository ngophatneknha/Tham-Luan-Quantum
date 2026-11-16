"""Benchmark mo phong mach luong tu: cung mach, nhieu bo mo phong. Moi lan do = 1 tien trinh rieng."""
import sys, json, time, subprocess, os
import numpy as np

D = 10  # so lop


def angles(n):
    return np.random.default_rng(123).uniform(0, 2 * np.pi, (D, n))


# ---------- worker ----------
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
    pr = (psi.real ** 2 + psi.imag ** 2)
    h = 2 ** (n - 1)
    return float(pr[:h].sum() - pr[h:].sum())


def make(sim, n):
    th = angles(n)
    if sim in ("numpy", "cupy64", "cupy128"):
        if sim == "numpy":
            return lambda: sim_array(np, np.complex128, n)
        import cupy as cp
        dt = cp.complex64 if sim == "cupy64" else cp.complex128

        def f():
            r = sim_array(cp, dt, n); cp.cuda.Stream.null.synchronize(); return r
        return f
    if sim.startswith("pl_"):
        import pennylane as qml
        dev = qml.device("default.qubit" if sim == "pl_default" else "lightning.qubit", wires=n)

        @qml.qnode(dev)
        def circ():
            for l in range(D):
                for k in range(n): qml.RY(th[l, k], wires=k)
                for c in range(n - 1): qml.CNOT(wires=[c, c + 1])
            return qml.expval(qml.PauliZ(0))
        return lambda: float(circ())
    if sim == "aer":
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(n)
        for l in range(D):
            for k in range(n): qc.ry(float(th[l, k]), k)
            for c in range(n - 1): qc.cx(c, c + 1)
        qc.save_expectation_value(SparsePauliOp("I" * (n - 1) + "Z"), list(range(n)))
        be = AerSimulator(method="statevector")
        return lambda: float(be.run(qc).result().data(0)["expectation_value"])
    raise ValueError(sim)


def worker(sim, n):
    import psutil
    make(sim, 4)()  # khoi dong
    f = make(sim, n)
    reps = 3 if n <= 18 else 1
    ts, val = [], None
    for _ in range(reps):
        t0 = time.perf_counter(); val = f(); ts.append(time.perf_counter() - t0)
    mem = psutil.Process().memory_info().peak_wset / 2 ** 20
    vram = None
    if sim.startswith("cupy"):
        import cupy as cp
        vram = cp.get_default_memory_pool().total_bytes() / 2 ** 20
    print(json.dumps({"sim": sim, "n": n, "sec": float(np.median(ts)), "val": val, "ram_mb": mem, "vram_mb": vram}))


# ---------- driver ----------
def driver():
    import psutil
    avail = psutil.virtual_memory().available / 2 ** 20
    info = {"avail_ram_mb_at_start": avail, "total_ram_mb": psutil.virtual_memory().total / 2 ** 20}
    out = {"info": info, "runs": []}
    sims = ["numpy", "pl_default", "pl_lightning", "aer", "cupy64", "cupy128"]
    for sim in sims:
        for n in range(10, 33, 2):
            est_state = 16 * 2 ** n / 2 ** 20  # MiB cho complex128
            gpu = sim.startswith("cupy")
            if gpu:
                per = 8 if sim == "cupy64" else 16
                if per * 2 ** n * 2.5 / 2 ** 20 > 3000: print("skip(vram)", sim, n, flush=True); break
            elif est_state * 3 > avail * 0.85:
                print("skip(ram)", sim, n, flush=True); break
            try:
                r = subprocess.run([sys.executable, __file__, "worker", sim, str(n)], capture_output=True, text=True, timeout=240,
                                   env={**os.environ, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "")} )
            except subprocess.TimeoutExpired:
                print("timeout", sim, n, flush=True); out["runs"].append({"sim": sim, "n": n, "status": "timeout"}); break
            if r.returncode != 0 or not r.stdout.strip():
                print("fail", sim, n, r.stderr[-300:].replace("\n", " "), flush=True)
                out["runs"].append({"sim": sim, "n": n, "status": "fail"}); break
            rec = json.loads(r.stdout.strip().splitlines()[-1]); rec["status"] = "ok"
            out["runs"].append(rec)
            print(sim, n, "%.3fs" % rec["sec"], "val=%.5f" % rec["val"], "ram=%.0fMB" % rec["ram_mb"], flush=True)
            json.dump(out, open("bench.json", "w"), indent=1)
            if rec["sec"] > 90: break
    json.dump(out, open("bench.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1:2] == ["worker"]:
        worker(sys.argv[2], int(sys.argv[3]))
    else:
        driver()
