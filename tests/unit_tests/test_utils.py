import os
import sys
import subprocess
import signal


def run_torchrun(script_path, args):
    """
    Run a PyTorch distributed script using torchrun with proper cleanup.

    Args:
        script_path (str): Path to the script to run
        args (argparse.Namespace): Command line arguments to pass to the script

    Raises:
        subprocess.CalledProcessError: If the process exits with a non-zero status
    """
    ngpu = args.ngpu if hasattr(args, 'ngpu') else int(os.environ.get("NGPU", 1))
    log_rank = os.environ.get("LOG_RANK", "0")
    current_env = os.environ.copy()

    # Set environment variable to help with cleanup
    current_env["PYTHONUNBUFFERED"] = "1"  # Ensure output is not buffered

    cmd_args = []
    for k, v in vars(args).items():
        if k == 'run_test':
            continue
        elif isinstance(v, bool) and v is True:
            # For boolean flags that are True, just add the flag without a value
            cmd_args.append(f"--{k}")
        elif v is not None:
            # For all other arguments, add with their values
            cmd_args.append(f"--{k}={v}")

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
          ] + cmd_args

    # Use a process group so signals are propagated to children
    try:
        process = subprocess.Popen(
            cmd,
            env=current_env,
            start_new_session=True  # Create a new process group for easier cleanup
        )

        # Wait for completion and check return code
        return_code = process.wait()
        if return_code != 0:
            print(f"Command failed with exit code {return_code}")
            print(f"Command: {' '.join(cmd)}")
            raise subprocess.CalledProcessError(return_code, cmd)

    except KeyboardInterrupt:
        # On keyboard interrupt, kill the entire process group
        print("Received interrupt, terminating processes...")
        if process.poll() is None:  # Process is still running
            try:
                # Send signal to the process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)  # Give some time to exit gracefully
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        raise