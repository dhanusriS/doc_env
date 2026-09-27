"""Educational raw-key distillation for the PCS-CV-QKD dashboard.

Reconciliation is modeled for demonstration purposes; this is not a validated
LDPC or composable privacy-amplification implementation.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from dataclasses import dataclass

import numpy as np

from core.channel import OpticalChannel
from core.constellation import PCSConstellation


@dataclass
class DistillationResult:
    symbols: int
    bits_per_symbol: int
    raw_bits: int
    mismatched_symbols: int
    raw_qber: float
    reconciliation_leak_bits: int
    final_key_bits: int
    final_key_base64: str


def _indices_to_bits(indices: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.uint8)
    return ((indices[:, None].astype(np.uint64) >> shifts) & 1).astype(np.uint8).ravel()


def _extract(bits: np.ndarray, output_bits: int) -> tuple[np.ndarray, bytes]:
    output_bits = min(max(int(output_bits), 0), int(bits.size))
    seed = secrets.token_bytes(32)
    if output_bits == 0:
        return np.empty(0, dtype=np.uint8), seed
    padding = (-bits.size) % 8
    payload = np.packbits(np.pad(bits, (0, padding))).tobytes()
    digest = hashlib.shake_256(seed + payload).digest((output_bits + 7) // 8)
    return np.unpackbits(np.frombuffer(digest, dtype=np.uint8))[:output_bits], seed


def simulate_key_distillation(
    constellation: PCSConstellation,
    channel: OpticalChannel,
    distance_km: float,
    nu: float,
    symbols: int = 2_000,
    secure_bits_requested: int = 128,
    xi_override: float | None = None,
    reconciliation_efficiency: float = 0.90,
) -> DistillationResult:
    """Run simulated decoding, modeled reconciliation, and extraction."""
    if symbols < 1:
        raise ValueError("symbols must be positive")
    if constellation.M & (constellation.M - 1):
        raise ValueError("constellation size must be a power of two")

    bits_per_symbol = int(np.log2(constellation.M))
    x_alice, p_alice, alice_indices = constellation.draw_symbols(symbols, nu)
    x_bob, p_bob, _ = channel.transmit(x_alice, p_alice, distance_km, xi_override)
    distances = (
        (x_bob[:, None] - constellation.points_x[None, :]) ** 2
        + (p_bob[:, None] - constellation.points_p[None, :]) ** 2
    )
    bob_indices = np.argmin(distances, axis=1)
    mismatches = int(np.count_nonzero(alice_indices != bob_indices))
    alice_bits = _indices_to_bits(alice_indices, bits_per_symbol)
    raw_bits = int(alice_bits.size)
    leak_bits = int(np.ceil((1.0 - reconciliation_efficiency) * raw_bits))
    reconciled_bits = alice_bits.copy()
    final_bits, _ = _extract(
        reconciled_bits,
        min(int(secure_bits_requested), max(raw_bits - leak_bits, 0)),
    )
    return DistillationResult(
        symbols=symbols,
        bits_per_symbol=bits_per_symbol,
        raw_bits=raw_bits,
        mismatched_symbols=mismatches,
        raw_qber=mismatches / symbols,
        reconciliation_leak_bits=leak_bits,
        final_key_bits=int(final_bits.size),
        final_key_base64=base64.b64encode(np.packbits(np.pad(final_bits, (0, (-final_bits.size) % 8))).tobytes()).decode("ascii") if final_bits.size else "",
    )
