#!/usr/bin/env python3
import time
import torch

def main(run_minutes: float = 30.0, matrix_size: int = 8192):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Ensure a GPU is present and PyTorch is built with CUDA.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True  # optimize kernels for the shapes we use

    # Create large tensors once; reuse to avoid host-device transfer overhead
    # Adjust matrix_size if you run out of memory; 4096 or 6144 are safer on smaller GPUs.
    a = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    b = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)
    c = torch.randn((matrix_size, matrix_size), device=device, dtype=torch.float32)

    # Optional: a simple CNN to add variety of workloads (conv, activation, matmul)
    conv = torch.nn.Conv2d(
        in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False
    ).to(device)
    x = torch.randn((32, 64, 256, 256), device=device, dtype=torch.float32)

    # Warm-up to ensure kernels are compiled and GPU clocks are up
    for _ in range(5):
        _ = torch.matmul(a, b)
        _ = torch.relu(conv(x))
    torch.cuda.synchronize()

    end_time = time.time() + run_minutes * 60.0
    i = 0

    # Continuous workload loop
    while time.time() < end_time:
        # Heavy GEMM operations (matrix multiplications)
        y1 = torch.matmul(a, b)
        y2 = torch.matmul(b, c)
        y3 = torch.matmul(c, a)

        # Elementwise ops to keep ALUs busy
        y1 = torch.sin(y1) + torch.cos(y2)
        y2 = torch.tanh(y3) * y1

        # Convolutional workload
        y4 = conv(x)
        y4 = torch.relu(y4)
        y4 = y4 * 1.0001 - 0.0001  # minor compute to prevent fusion into a no-op

        # Prevent lazy evaluation and ensure kernels finish before the next loop
        # (PyTorch schedules ops asynchronously; synchronize to sustain steady load)
        torch.cuda.synchronize()

        # Free intermediate references to avoid memory growth
        del y1, y2, y3, y4

        # Optional tiny pause to modulate thermals; set to 0 for max load
        # time.sleep(0.0)

        i += 1
        # Lightweight progress feedback (doesn't save anything)
        if i % 50 == 0:
            # Printing is fine; remove if you want absolutely zero I/O
            print(f"Iterations: {i}, elapsed: {int((time.time() - (end_time - run_minutes*60)))}s")

    print("Completed GPU run.")

if __name__ == "__main__":
    # You can edit these values or pass via environment/arguments if you extend the script.
    main(run_minutes=30.0, matrix_size=8192)