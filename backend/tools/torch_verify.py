"""Minimal helper to report torch info for the backend env."""

from __future__ import annotations

if __name__ == "__main__":
    try:
        import torch
    except ModuleNotFoundError:
        print("Torch is not installed in the backend environment.")
        print("Run 'make torch-install-cpu' or 'make torch-install-cuda' first.")
        raise SystemExit(1) from None

    print(f"torch.__version__: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    cuda_devices = torch.cuda.device_count()
    print(f"torch.cuda.device_count(): {cuda_devices}")
    if cuda_available and cuda_devices:  # pragma: no cover - best effort check
        print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    if cuda_available:
        print(f"torch.version.cuda: {torch.version.cuda}")
