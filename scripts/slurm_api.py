#!/usr/bin/env python3
"""Read-only client for Slurm REST API status queries."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request


def get_json(
    base_url: str,
    endpoint: str,
    token: str,
    user: str | None,
    auth_mode: str = "bearer",
) -> object:
    if not endpoint.lstrip("/").startswith("slurm/"):
        raise ValueError("endpoint must be below /slurm/; mutations are not supported")
    url = urllib.parse.urljoin(base_url.rstrip("/") + "/", endpoint.lstrip("/"))
    headers = {"Accept": "application/json"}
    if auth_mode == "bearer":
        headers["Authorization"] = f"Bearer {token}"
    elif auth_mode == "slurm":
        headers["X-SLURM-USER-TOKEN"] = token
    else:
        raise ValueError(f"unsupported auth mode: {auth_mode}")
    if user and auth_mode == "slurm":
        headers["X-SLURM-USER-NAME"] = user
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read(1000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Slurm API GET failed ({exc.code}): {body}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--endpoint", default="slurm/v0.0.41/jobs")
    parser.add_argument("--auth-mode", choices=["bearer", "slurm"], default="bearer")
    parser.add_argument("--token-env", default="SLURM_JWT")
    parser.add_argument("--user", default=os.environ.get("USER"))
    parser.add_argument("--output")
    args = parser.parse_args()
    token = os.environ.get(args.token_env)
    if not token:
        parser.error(f"environment variable {args.token_env} is not set")
    result = get_json(args.base_url, args.endpoint, token, args.user, args.auth_mode)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
