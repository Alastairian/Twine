import torch
from pathos_core import PathosMatrixOrchestrator
from lagos_marix import LogosMatrixOrchestrator

def test_pathos_works():
    pathos = PathosMatrixOrchestrator(sensory_input_size=10)
    data = torch.rand(1, 10)
    result = pathos.run_intuitive_analysis(data)
    assert result is not None

def test_logos_works():
    logos = LogosMatrixOrchestrator()
    data = torch.rand(1, 10)
    result = logos.run_logical_analysis(data)
    assert result is not None