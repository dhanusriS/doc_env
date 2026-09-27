"""Tests for the educational Eve attack simulation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.channel import ChannelConfig, OpticalChannel
from core.constellation import PCSConstellation
from core.key_rate import optimal_nu
from simulation.eve_attack import run_eve_attack_experiment



def _run(probability: float):
    constellation = PCSConstellation(256)
    channel = OpticalChannel(
        ChannelConfig(excess_noise=0.005, n_wdm=0, nu_el=0.01)
    )
    return run_eve_attack_experiment(
        constellation=constellation,
        channel=channel,
        distance_km=30,
        nu=7.22,
        attack_probability=probability,
        symbols=20_000,
        seed=2026,
        xi_override=0.005,
    )


def test_attack_off_has_no_sample_mutation():
    result = _run(0.0)
    assert result.attacked_symbols == 0
    assert result.attack_noise == 0.0


def test_attack_strength_changes_observed_statistics():
    results = [_run(probability) for probability in [0.0, 0.1, 0.5, 1.0]]
    assert [result.attacked_symbols for result in results] == [0, 1912, 9990, 20000]
    assert results[0].xi_max < results[1].xi_max < results[2].xi_max < results[3].xi_max
    assert results[0].T_min > results[1].T_min > results[2].T_min > results[3].T_min


def test_attack_reaches_finite_size_skr():
    constellation = PCSConstellation(256)
    values = []
    for probability in [0.0, 0.1, 0.5, 1.0]:
        result = _run(probability)
        nu, skr = optimal_nu(
            result.T_min,
            result.xi_max,
            0.01,
            constellation,
            100_000_000,
            n_grid=120,
        )
        values.append(skr.skr_finite)
    assert values[0] > values[1] > values[2] > values[3]
    assert values[0] > 0.0
    assert values[2] < 0.0
    assert values[3] < 0.0


def test_discrete_kl_is_nonnegative():
    constellation = PCSConstellation(256)
    for nu in [0.01, 2.0, 7.22, 20.0]:
        assert constellation.kl_from_gaussian(nu) >= -1e-12
