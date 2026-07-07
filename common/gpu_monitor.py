import time
import pynvml

def daemon_process(time_interval, out_path):
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        while True:
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory = mem.used / 1024 / 1024  # MiB

            with open(out_path, "a") as f:
                f.write(str(gpu_memory) + "\n")

            time.sleep(time_interval)
    finally:
        pynvml.nvmlShutdown()