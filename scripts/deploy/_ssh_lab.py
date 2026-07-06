"""Tiny SSH runner for the lab server."""
import sys
import paramiko

HOST = "10.87.82.105"
PORT = 22
USER = "zjulab"
PASS = "zjucst"


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
    while chan.recv_ready():
        out += chan.recv(65536)
    while chan.recv_stderr_ready():
        out += chan.recv_stderr(65536)
    code = chan.recv_exit_status()
    sys.stdout.write(out.decode("utf-8", "replace"))
    cli.close()
    return code


if __name__ == "__main__":
    # Take all args as the command (ignore timeout parameter issues)
    if len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
    else:
        cmd = "echo connected"
    sys.exit(run(cmd, 600))
