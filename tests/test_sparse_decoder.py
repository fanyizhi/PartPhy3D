from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tactile3d.research.sparse_decoder import (
    decode_surface_weights,
    sparse_weight_decoder,
)


def test_sparse_weight_values_equal_dense_only_on_active_cells() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    features = torch.randn(1, 3, 24, 2)
    features[0, :, :8] = 2
    features[0, :, 16:] = -2
    decoder = SimpleNamespace(
        training=False,
        net_sdf=lambda x: x[..., :1],
        net_deformation=torch.nn.Linear(6, 3),
        net_weight=torch.nn.Linear(48, 21),
    )
    cubes = torch.arange(24).reshape(3, 8)
    stats = []
    sdf, deformation, weights = decode_surface_weights(decoder, features, cubes, stats)
    flat = features.permute(0, 2, 1, 3).reshape(1, 24, 6)
    dense = decoder.net_weight(flat[:, cubes.reshape(-1)].reshape(1, 3, 48)) * 0.1
    active = ((sdf[0, :, 0] < 0)[cubes].sum(1) > 0) & (
        (sdf[0, :, 0] < 0)[cubes].sum(1) < 8
    )
    torch.testing.assert_close(weights[:, active], dense[:, active])
    torch.testing.assert_close(deformation, decoder.net_deformation(flat))
    np.testing.assert_equal(stats[0]["evaluated_weight_cells"], 1)
    with pytest.raises(ValueError, match="无零交叉"):
        decode_surface_weights(decoder, torch.ones_like(features), cubes, [])


def test_decoder_context_restores_original_method_on_failure() -> None:
    decoder = SimpleNamespace(get_geometry_prediction="original")
    model = SimpleNamespace(synthesizer=SimpleNamespace(decoder=decoder))
    with pytest.raises(RuntimeError, match="expected"), sparse_weight_decoder(model):
        assert callable(decoder.get_geometry_prediction)
        raise RuntimeError("expected")
    assert decoder.get_geometry_prediction == "original"
