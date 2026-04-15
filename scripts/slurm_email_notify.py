#!/usr/bin/env python3
from __future__ import annotations

import os
import smtplib
import socket
from email.message import EmailMessage


def _env_flag(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw is None:
    return default
  return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
  to_addr = os.environ.get("MJLAB_SLURM_NOTIFY_TO")
  smtp_host = os.environ.get("MJLAB_SMTP_HOST")
  dry_run = _env_flag("MJLAB_SLURM_NOTIFY_DRY_RUN", default=False)
  if not to_addr:
    return 0
  if not smtp_host and not dry_run:
    return 0

  smtp_port = int(os.environ.get("MJLAB_SMTP_PORT", "587"))
  from_addr = os.environ.get("MJLAB_SLURM_NOTIFY_FROM") or os.environ.get(
    "MJLAB_SMTP_USERNAME"
  )
  if not from_addr:
    raise ValueError(
      "Set MJLAB_SLURM_NOTIFY_FROM or MJLAB_SMTP_USERNAME when email notify is enabled."
    )

  job_id = os.environ.get("SLURM_JOB_ID", "unknown")
  job_name = os.environ.get("SLURM_JOB_NAME", "mjlab")
  node_name = (
    os.environ.get("SLURMD_NODENAME")
    or os.environ.get("SLURM_NODELIST")
    or socket.gethostname()
  )
  command = os.environ.get("MJLAB_SLURM_NOTIFY_COMMAND", "")
  cwd = os.environ.get("MJLAB_SLURM_NOTIFY_CWD", os.getcwd())
  runtime_root = os.environ.get("MJLAB_RUNTIME_ROOT", "")

  subject = os.environ.get("MJLAB_SLURM_NOTIFY_SUBJECT") or (
    f"[mjlab] Slurm job {job_id} started on {node_name}"
  )

  body = "\n".join(
    [
      "mjlab Slurm job started.",
      "",
      f"Job ID: {job_id}",
      f"Job Name: {job_name}",
      f"Node: {node_name}",
      f"Host: {socket.gethostname()}",
      f"Working Directory: {cwd}",
      f"Runtime Root: {runtime_root or '(unset)'}",
      f"Command: {command or '(unset)'}",
    ]
  )

  message = EmailMessage()
  message["Subject"] = subject
  message["From"] = from_addr
  message["To"] = to_addr
  message.set_content(body)

  if dry_run:
    print("=== MJLAB Slurm email dry run ===")
    print(f"Subject: {subject}")
    print(f"From: {from_addr}")
    print(f"To: {to_addr}")
    print("")
    print(body)
    return 0

  username = os.environ.get("MJLAB_SMTP_USERNAME")
  password = os.environ.get("MJLAB_SMTP_PASSWORD")
  use_ssl = _env_flag("MJLAB_SMTP_SSL", default=False)
  use_starttls = _env_flag("MJLAB_SMTP_STARTTLS", default=not use_ssl)

  if use_ssl:
    with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
      if username:
        server.login(username, password or "")
      server.send_message(message)
    return 0

  with smtplib.SMTP(smtp_host, smtp_port) as server:
    if use_starttls:
      server.starttls()
    if username:
      server.login(username, password or "")
    server.send_message(message)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
