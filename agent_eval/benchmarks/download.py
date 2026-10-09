"""Download benchmark data with checksum verification.

Nothing large is committed to the repository. `agent-eval download` fetches
each dataset into the data directory (AGENT_EVAL_DATA_DIR, default data/),
verifies its SHA-256 against the value pinned below and its record count, and,
for HotpotQA, writes the fixed 200-question subset used by the benchmark.

A dataset whose checksum is not pinned yet is still verified by record count;
its SHA-256 is printed and written to <data_dir>/CHECKSUMS so it can be pinned
in DATASETS (and committed) after the first download.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from ..logs import get_logger
from . import hotpotqa

log = get_logger(__name__)


class DownloadError(RuntimeError):
    """A download failed, or the file did not match its pinned checksum or record count."""


@dataclass(frozen=True)
class Dataset:
    """Where a dataset comes from and what a correct copy looks like."""

    name: str
    url: str
    filename: str
    records: int
    sha256: str | None = None


DATASETS: dict[str, Dataset] = {
    "humaneval": Dataset(
        "humaneval",
        "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
        "humaneval/HumanEval.jsonl.gz",
        records=164,
        sha256="b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef",
    ),
    "hotpotqa": Dataset(
        "hotpotqa",
        "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
        "hotpotqa/hotpot_dev_distractor_v1.json",
        records=7405,
        sha256=None,  # not pinned yet: the build environment could not reach the host
    ),
}

HOTPOT_SUBSET = f"hotpotqa/dev_distractor_{hotpotqa.SUBSET_SIZE}_seed{hotpotqa.SUBSET_SEED}.jsonl"

Opener = Callable[[str], IO[bytes]]


def _default_opener(url: str) -> IO[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "agent-eval-harness"})
    response: IO[bytes] = urllib.request.urlopen(req, timeout=120)
    return response


def sha256_of(path: Path) -> str:
    """SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_records(path: Path) -> int:
    """Records in a JSON array, JSONL or gzipped JSONL file."""
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    if path.suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    return len(data)


def _record_checksum(data_dir: Path, rel: str, digest: str) -> None:
    sums = data_dir / "CHECKSUMS"
    lines = [ln for ln in (sums.read_text().splitlines() if sums.exists() else []) if not ln.endswith(f"  {rel}")]
    lines.append(f"{digest}  {rel}")
    sums.write_text("\n".join(sorted(lines)) + "\n")


def fetch(ds: Dataset, data_dir: Path, opener: Opener = _default_opener, force: bool = False) -> Path:
    """Download `ds` into `data_dir` (unless a verified copy is there) and verify it."""
    target = data_dir / ds.filename
    if target.exists() and not force and (ds.sha256 is None or sha256_of(target) == ds.sha256):
        log.info("dataset already present", extra={"fields": {"dataset": ds.name, "path": str(target)}})
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        try:
            with opener(ds.url) as src, open(part, "wb") as dst:
                shutil.copyfileobj(src, dst)
        except OSError as exc:
            part.unlink(missing_ok=True)
            raise DownloadError(f"could not download {ds.name} from {ds.url}: {exc}") from exc
        digest = sha256_of(part)
        if ds.sha256 is not None and digest != ds.sha256:
            part.unlink()
            raise DownloadError(f"{ds.name}: SHA-256 mismatch (expected {ds.sha256}, got {digest})")
        part.replace(target)
    digest = sha256_of(target)
    if ds.sha256 is None:
        log.warning(
            "checksum not pinned",
            extra={"fields": {"dataset": ds.name, "sha256": digest, "hint": "pin it in benchmarks/download.py"}},
        )
    _record_checksum(data_dir, ds.filename, digest)
    found = count_records(target)
    if found != ds.records:
        raise DownloadError(f"{ds.name}: expected {ds.records} records, found {found}")
    return target


def write_hotpot_subset(source: Path, data_dir: Path) -> Path:
    """Write the fixed seeded subset of HotpotQA dev questions as JSONL."""
    records = json.loads(source.read_text(encoding="utf-8"))
    subset = hotpotqa.select_subset(records)
    out = data_dir / HOTPOT_SUBSET
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in subset), encoding="utf-8")
    _record_checksum(data_dir, HOTPOT_SUBSET, sha256_of(out))
    return out


def download(
    names: list[str], data_dir: str | Path, opener: Opener = _default_opener, force: bool = False
) -> list[Path]:
    """Fetch and verify the named datasets ("all" for every one); returns the files ready to use."""
    data_dir = Path(data_dir)
    wanted = list(DATASETS) if "all" in names else names
    unknown = [n for n in wanted if n not in DATASETS]
    if unknown:
        raise ValueError(f"unknown dataset(s): {', '.join(unknown)} (choose from: {', '.join(DATASETS)}, all)")
    ready = []
    for name in wanted:
        path = fetch(DATASETS[name], data_dir, opener, force)
        if name == "hotpotqa":
            path = write_hotpot_subset(path, data_dir)
        ready.append(path)
    return ready
