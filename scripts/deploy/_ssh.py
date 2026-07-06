#!/usr/bin/env python
"""Tiny SSH runner for the AutoDL box (paramiko, password auth)."""
import sys
import paramiko

HOST = "connect.bjb1.seetacloud.com"
PORT = 27924
USER = "root"
PASS = "F5PcLR9FufOe"


def run(cmd: str, timeout: int = 600) -> int:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(HOST, port=PORT, username=USER, password=PASS, timeout=30)
    chan = cli.get_transport().open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(65536)
        if chan.recv_stderr_ready():
            out += chan.recv_stderr(65536)
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
    # drain
    while chan.recv_ready():
        out += chan.recv(65536)
    while chan.recv_stderr_ready():
        out += chan.recv_stderr(65536)
    code = chan.recv_exit_status()
    sys.stdout.write(out.decode("utf-8", "replace"))
    cli.close()
    return code


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo connected"
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    sys.exit(run(cmd, t))
