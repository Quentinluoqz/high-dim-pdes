#!/usr/bin/env python3
"""Collect Slurm accounting records and join array tasks to a run manifest."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import TextIO

SACCT_FIELDS = [
    "JobIDRaw",
    "JobName",
    "State",
    "ElapsedRaw",
    "MaxRSS",
    "ExitCode",
    "AllocTRES",
    "Partition",
    "QOS",
    "Submit",
    "Start",
    "End",
]


def _read_sacct(handle: TextIO) -> list[dict[str, str]]:
    return list(csv.DictReader(handle, delimiter="|"))


def _array_index(job_id: str) -> int | None:
    # Ignore .batch/.extern steps; their resource data stays in the parent record.
    if "." in job_id:
        return None
    match = re.search(r"_(\d+)$", job_id)
    return int(match.group(1)) if match else None


def merge_ledger(
    manifest_rows: list[dict[str, str]],
    accounting_rows: list[dict[str, str]],
    job_prefix: str,
) -> tuple[list[str], list[dict[str, str]]]:
    manifest_by_index = {int(row["array_index"]): row for row in manifest_rows}
    output: list[dict[str, str]] = []
    sacct_names = [f"sacct_{field.lower()}" for field in SACCT_FIELDS]
    for record in accounting_rows:
        if not record.get("JobName", "").startswith(job_prefix):
            continue
        index = _array_index(record.get("JobIDRaw", ""))
        if index is None:
            continue
        merged = dict(manifest_by_index.get(index, {}))
        merged["manifest_match"] = "true" if index in manifest_by_index else "false"
        for field, destination in zip(SACCT_FIELDS, sacct_names, strict=True):
            merged[destination] = record.get(field, "")
        output.append(merged)
    fieldnames = list(manifest_rows[0]) if manifest_rows else []
    fieldnames += ["manifest_match", *sacct_names]
    return fieldnames, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--job-prefix", default="hdpde-")
    parser.add_argument("--output", type=Path, default=Path("ledger.csv"))
    parser.add_argument("--sacct-file", type=Path, help="use saved pipe-delimited sacct output")
    parser.add_argument("--start-time", help="optional sacct -S value")
    args = parser.parse_args()

    with args.manifest.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if args.sacct_file:
        with args.sacct_file.open(newline="", encoding="utf-8") as handle:
            accounting = _read_sacct(handle)
    else:
        command = ["sacct", "-X", "-P", "-n", "-o", ",".join(SACCT_FIELDS)]
        # Headerless output is less robust; explicitly prepend the known header.
        if args.start_time:
            command.extend(("-S", args.start_time))
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        content = "|".join(SACCT_FIELDS) + "\n" + completed.stdout
        accounting = list(csv.DictReader(content.splitlines(), delimiter="|"))
    fields, ledger = merge_ledger(manifest, accounting, args.job_prefix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ledger)
    print(f"wrote {len(ledger)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
