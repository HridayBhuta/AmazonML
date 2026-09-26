#!/usr/bin/env python3
"""Amazon ML Challenge 2026 - Business Entity Resolution: one entry point for everything.

Commands
  locate    find (or extract) the official student_resource folder and print its path
  smoke     build a tiny sample from the real data and run the whole pipeline on it (~3-5 min)
  run       full pipeline for one experiment:  run --name baseline [--max-df 400 --leaves 63 ...]
  sweep     run every experiment in configs/sweep.json (skips ones already recorded)
  compare   rank recorded experiments by untouched holdout F0.5 and select the best
  finalize  re-validate on the real data, fill Documentation_template.md, build <team>_submission.zip
  prune     delete cache folders not used by the selected best run (frees disk)
  auto      smoke -> baseline -> finalize baseline -> sweep -> compare -> finalize best
"""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er.common import Config  # noqa: E402

REQUIRED = ["dataset/train/train_source1.tsv", "dataset/train/train_source2.tsv",
            "dataset/train/train_source3.tsv", "dataset/train/train_ground_truth.tsv",
            "dataset/test/test_source1.tsv", "dataset/test/test_source2.tsv", "dataset/test/test_source3.tsv",
            "utils/validate_submission.py", "Documentation_template.md"]
DATA_DIR = ROOT / "data"
CONFIG_FIELDS = {f.name for f in dataclasses.fields(Config)}


def _is_sr(p: Path) -> bool:
    try:
        ok = all((p / r).is_file() for r in REQUIRED)
    except PermissionError:
        return False
    if ok and DATA_DIR.resolve() in p.resolve().parents:   # extracted by us: must be complete
        return (DATA_DIR / ".extract_complete").is_file()
    return ok


def _members(z: zipfile.ZipFile):
    for m in z.infolist():
        n = m.filename
        if n.startswith("__MACOSX") or n.endswith(".DS_Store") or n.endswith("/"):
            continue
        yield m


def _zip_has_data(zpath: Path) -> bool:
    try:
        with zipfile.ZipFile(zpath) as z:
            names = [m.filename for m in _members(z)]
    except (zipfile.BadZipFile, OSError):
        return False
    return any(n.endswith(r) for n in names for r in REQUIRED[:7])


def _safe_extract(zips: list[Path]) -> Path | None:
    """Extract one or several (Drive multi-part) zips into data/, atomically per file."""
    dest = DATA_DIR.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    marker = DATA_DIR / ".extract_complete"
    if marker.exists():
        marker.unlink()
    for zpath in zips:
        print(f"extracting {zpath} -> {dest}")
        with zipfile.ZipFile(zpath) as z:
            for m in _members(z):
                target = (dest / m.filename).resolve()
                if dest not in target.parents:
                    raise ValueError(f"unsafe path in zip: {m.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file() and target.stat().st_size == m.file_size:
                    continue
                part = target.with_name(target.name + ".part")
                with z.open(m) as src, open(part, "wb") as out:
                    shutil.copyfileobj(src, out, 1 << 22)
                os.replace(part, target)
    marker.write_text("ok")
    for cand in [dest / "student_resource", dest] + sorted(dest.glob("*/student_resource")) + sorted(dest.glob("*")):
        if cand.is_dir() and _is_sr(cand):
            return cand
    return None


def _merge_split_folders(dirs: list[Path]) -> Path | None:
    """Safari auto-unzips a two-part Drive download into 'student_resource' and 'student_resource 2'."""
    found = {}
    for d in dirs:
        for r in REQUIRED:
            for c in (d / r, d / "student_resource" / r):
                if r not in found and c.is_file():
                    found[r] = c
    if len(found) != len(REQUIRED):
        return None
    tgt = DATA_DIR / "student_resource"
    for r, src in found.items():
        dst = tgt / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not (dst.is_file() and dst.stat().st_size == src.stat().st_size):
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    (DATA_DIR / ".extract_complete").write_text("ok")
    return tgt


def locate(explicit: str | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        for c in (p, p / "student_resource"):
            if _is_sr(c):
                return c
        if p.suffix == ".zip" and p.is_file():
            got = _safe_extract([p])
            if got:
                return got.resolve()
        missing = [r for r in REQUIRED if not (p / r).is_file()]
        raise SystemExit(f"--data {p} is not a complete student_resource folder; missing: {missing}")
    home = Path.home()
    bases = [ROOT, ROOT.parent, home / "Downloads", home / "Desktop", home / "Documents"]
    dirs = [DATA_DIR / "student_resource", DATA_DIR]
    try:
        for b in bases:
            if b.is_dir():
                dirs += sorted(b.glob("*student_resource*")) + sorted(b.glob("drive-download-*"))
                dirs += sorted(b.glob("*student_resource*/student_resource"))
    except PermissionError:
        print("macOS blocked access to Downloads/Desktop/Documents. Allow Terminal in System Settings > "
              "Privacy & Security > Files and Folders, or move the data into this folder.")
    for d in dirs:
        if d.is_dir():
            for c in (d, d / "student_resource"):
                if _is_sr(c):
                    return c.resolve()
    merged = _merge_split_folders([d for d in dirs if d.is_dir()])
    if merged:
        return merged.resolve()
    zips = []
    for b in bases:
        if b.is_dir():
            try:
                zips += sorted(b.glob("*student_resource*.zip")) + sorted(b.glob("drive-download-*.zip"))
            except PermissionError:
                pass
    zips = [z for z in dict.fromkeys(zips) if _zip_has_data(z)]
    for z in zips:                       # a single complete zip
        got = _safe_extract([z])
        if got:
            return got.resolve()
    if len(zips) > 1:                    # Drive multi-part download
        got = _safe_extract(zips)
        if got:
            return got.resolve()
    raise SystemExit(
        "Could not find the official data. Put the 'student_resource' folder or the official zip "
        "(e.g. 'OFFICIAL_DATA_student_resource.zip', unopened and not renamed) in this folder or in "
        "~/Downloads, then run again.")


def team_info():
    p = ROOT / "team_config.json"
    if not p.is_file():
        return "", ""
    try:
        t = json.loads(p.read_text(encoding="utf-8"))
        return str(t.get("team_name", "")).strip(), str(t.get("team_members", "")).strip()
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"WARNING: team_config.json is not valid JSON ({e}); team name left blank. "
              'Expected: {"team_name": "My Team", "team_members": "A, B, C"}')
        return "", ""


def make_cfg(a, name: str, overrides: dict | None = None) -> Config:
    cfg = Config(data=str(locate(a.data)), run=str(ROOT / "runs" / name), cache=str(ROOT / "cache"))
    for f in CONFIG_FIELDS - {"data", "run", "cache"}:
        v = getattr(a, f, None)
        if v is not None:
            setattr(cfg, f, v)
    bad = set(overrides or {}) - (CONFIG_FIELDS - {"data", "run", "cache", "smoke"})
    if bad:
        raise ValueError(f"unknown override keys {sorted(bad)}; valid: {sorted(CONFIG_FIELDS)}")
    for k, v in (overrides or {}).items():
        setattr(cfg, k, v)
    return cfg


def run_one(a, name: str, overrides: dict | None = None, note: str = ""):
    from er.pipeline import Pipeline
    from er import report
    t0 = time.time()
    p = Pipeline(make_cfg(a, name, overrides))
    (p.run / "artifacts" / "caches.json").write_text(json.dumps(
        {"norm": p.norm.name, "block": p.bdir.name, "feat": p.fdir.name}))
    p.all()
    row = report.record(p.run, name, note or json.dumps(overrides or {}))
    p.log(f"experiment {name} recorded in {report.RESULTS} ({(time.time() - t0) / 60:.1f} min): {json.dumps(row)}")
    return p.run


def cmd_finalize(a, run_name: str | None = None):
    from er import report
    if not run_name:
        best = report.compare(select=True)
        if not best:
            raise SystemExit("nothing to finalize")
        run_name = best["run"]
    run = ROOT / "runs" / run_name
    team, members = team_info()
    sr = locate(a.data)
    info = report.finalize(run, sr / "Documentation_template.md", team, members, ROOT / "final", data=sr)
    print(json.dumps(info, indent=2))
    print((ROOT / "final" / "SUBMIT_THESE.txt").read_text())
    return info


def cmd_smoke(a):
    from make_sample import make_sample
    from er.pipeline import Pipeline
    full = locate(a.data)
    sample = make_sample(full, ROOT / "cache" / "sample")
    cfg = Config(data=str(sample), run=str(ROOT / "runs" / "_smoke"), cache=str(ROOT / "cache" / "smoke"))
    for k, v in dict(smoke=True, chunk=2000, rounds=300, early_stop=30, tune_s1=100000, b_block=20000).items():
        setattr(cfg, k, v)
    if a.threads:
        cfg.threads = a.threads
    for d in (ROOT / "runs" / "_smoke", ROOT / "cache" / "smoke"):
        if d.exists():
            shutil.rmtree(d)
    p = Pipeline(cfg)
    res = p.all()
    m = json.loads((p.run / "artifacts" / "metrics.json").read_text())
    print("SMOKE TEST PASSED:", json.dumps({**res, "holdout_macro_f05(sample, not representative)":
                                             m["holdout_macro_f05_untouched"]}))


def cmd_sweep(a):
    from er import report
    plan = json.loads((ROOT / (a.plan or "configs/sweep.json")).read_text())
    done = set()
    if report.RESULTS.is_file():
        import polars as pl
        done = set(pl.read_csv(report.RESULTS, separator="\t", infer_schema=False)["run"].to_list())
    for exp in plan["experiments"]:
        if exp["name"] in done and not a.rerun:
            print(f"skip {exp['name']} (already recorded)")
            continue
        try:
            run_one(a, exp["name"], exp.get("overrides", {}), exp.get("why", ""))
        except Exception as e:  # keep sweeping; the failure is logged
            (ROOT / "experiments").mkdir(exist_ok=True)
            with open(ROOT / "experiments" / "failures.log", "a") as f:
                f.write(f"{time.strftime('%F %T')} {exp['name']}: {type(e).__name__}: {e}\n")
            print(f"experiment {exp['name']} FAILED: {e}")
    report.compare(select=True)


def cmd_prune():
    best = ROOT / "experiments" / "BEST.json"
    if not best.is_file():
        raise SystemExit("run `compare` first")
    run = json.loads(best.read_text())["run"]
    keep = set(json.loads((ROOT / "runs" / run / "artifacts" / "caches.json").read_text()).values())
    freed = 0
    for d in (ROOT / "cache").glob("*_*"):
        if d.is_dir() and d.name not in keep and d.name.split("_")[0] in ("block", "feat"):
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
    print(f"kept caches of best run '{run}': {sorted(keep)}; freed {freed / 2**30:.1f} GB")


def _preload_openmp():
    """If run.sh found a libomp inside scikit-learn (no Homebrew), load it before LightGBM."""
    lib = os.environ.get("ER_OMP_LIB")
    if lib and os.path.isfile(lib):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            print(f"could not preload {lib}: {e}")


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        os.nice(5)                       # stay polite; macOS strips DYLD vars from /usr/bin/nice
    except (AttributeError, OSError):
        pass
    _preload_openmp()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["locate", "smoke", "run", "sweep", "compare", "finalize", "prune", "auto"])
    ap.add_argument("--data", help="path to student_resource or the official zip (auto-detected when omitted)")
    ap.add_argument("--name", help="experiment/run name (run, finalize)")
    ap.add_argument("--plan", help="sweep plan json (default configs/sweep.json)")
    ap.add_argument("--rerun", action="store_true", help="sweep: rerun experiments already recorded")
    ap.add_argument("--note", default="")
    for f in dataclasses.fields(Config):
        if f.name in ("data", "run", "cache", "team", "members", "smoke"):
            continue
        flag = "--" + f.name.replace("_", "-")
        if str(f.type) in ("bool", "<class 'bool'>"):
            ap.add_argument(flag, dest=f.name, action="store_true", default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(str(f.type), str)
            ap.add_argument(flag, dest=f.name, type=typ, default=None)
    a = ap.parse_args()
    if a.name and (a.name.startswith("_") or "/" in a.name):
        raise SystemExit("run names must not start with '_' (reserved for smoke tests) or contain '/'")
    if a.command == "locate":
        print(locate(a.data))
    elif a.command == "smoke":
        cmd_smoke(a)
    elif a.command == "run":
        run_one(a, a.name or "baseline", note=a.note)
    elif a.command == "sweep":
        cmd_sweep(a)
    elif a.command == "compare":
        from er import report
        report.compare(select=True)
    elif a.command == "finalize":
        cmd_finalize(a, a.name)
    elif a.command == "prune":
        cmd_prune()
    elif a.command == "auto":
        cmd_smoke(a)
        run_one(a, "baseline", note="default configuration")
        cmd_finalize(a, "baseline")
        cmd_sweep(a)
        cmd_finalize(a)


if __name__ == "__main__":
    main()
