#!/usr/bin/env python3
"""Simulated S3 sync for model artifacts.

Usage:
  python scripts/s3_sync.py upload s3://bucket/path
  python scripts/s3_sync.py download s3://bucket/path

If boto3 is not available or credentials are missing, this script will print a stubbed message.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "backend" / "artifacts" / "model_bundle_v1"


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: s3_sync.py <upload|download> s3://bucket/path")
        return 1

    action = sys.argv[1]
    s3_uri = sys.argv[2]

    try:
        import boto3  # type: ignore
    except Exception:
        print(f"[stub] Would {action} {ARTIFACTS_DIR} <-> {s3_uri}")
        return 0

    s3 = boto3.client("s3")
    if not s3_uri.startswith("s3://"):
        print("Invalid S3 URI")
        return 1

    bucket, _, prefix = s3_uri[5:].partition("/")

    if action == "upload":
        for path in ARTIFACTS_DIR.rglob("*"):
            if path.is_file():
                key = f"{prefix}/{path.relative_to(ARTIFACTS_DIR)}".strip("/")
                s3.upload_file(str(path), bucket, key)
                print(f"Uploaded {path} to s3://{bucket}/{key}")
    elif action == "download":
        # minimal stub: list and download is not implemented for brevity
        print(f"[stub] Would download from s3://{bucket}/{prefix} to {ARTIFACTS_DIR}")
    else:
        print("Unknown action; use upload or download")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
