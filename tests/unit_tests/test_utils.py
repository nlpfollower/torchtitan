import os
import sys
import subprocess

def run_torchrun(script_path, args):
    ngpu = args.ngpu if hasattr(args, 'ngpu') else int(os.environ.get("NGPU", 1))
    log_rank = os.environ.get("LOG_RANK", "0")
    current_env = os.environ.copy()

    cmd = [
              sys.executable, "-m", "torch.distributed.run",
              f"--nproc_per_node={ngpu}",
              "--rdzv_backend=c10d",
              "--rdzv_endpoint=localhost:0",
              f"--local-ranks-filter={log_rank}",
              "--role=rank",
              "--tee=3",
              script_path,
              "--run_test"
          ] + [f"--{k}={v}" for k, v in vars(args).items() if k != 'run_test']

    try:
        subprocess.run(cmd, env=current_env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"torchrun command failed with exit code {e.returncode}")
        print(f"Command: {e.cmd}")
        print(f"Output: {e.output}")
        raise