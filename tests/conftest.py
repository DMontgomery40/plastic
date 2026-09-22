import pytest
import torch


def _available_devices() -> list[str]:
    out = ["cpu"]
    if torch.backends.mps.is_available():
        out.append("mps")
    if torch.cuda.is_available():
        out.append("cuda")
    return out


@pytest.fixture(params=_available_devices())
def device(request) -> torch.device:
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)
