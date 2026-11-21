"""Run independent game batches on the assigned GPUs for one learner."""
import os
import subprocess
import sys
import time


def run_workers(devices, task, checkpoint, folder, seed, progress, opponent=None, *, chunks=None):
    assert 1 <= len(devices) <= 6 and len(set(devices)) == len(devices)
    if chunks is None:
        chunks = max(4, len(devices))
    processes, logs = [], []
    try:
        for index, device in enumerate(devices):
            command = [sys.executable, '-u', '-m', 'scripts.selfplay_worker',
                       task, '--checkpoint', str(checkpoint), '--folder', str(folder),
                       '--worker', str(index), '--workers', str(len(devices)),
                       '--chunks', str(chunks), '--seed', str(seed + index)]
            if opponent is not None:
                pairs = 64 // len(devices) + int(index < 64 % len(devices))
                command.extend(['--opponent', str(opponent), '--pairs', str(pairs), '--total-pairs', '64'])
            log = (folder / f'{task}_worker{index}.log').open('a')
            logs.append(log)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(device), 'OMP_NUM_THREADS': '1',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            processes.append(subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT))
        while True:
            codes = [p.poll() for p in processes]
            progress()
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'{task} worker failed: {codes}; see worker logs in {folder}')
            if all(code == 0 for code in codes):
                return
            time.sleep(1)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
