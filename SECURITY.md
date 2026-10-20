# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| < 0.1 | No |

## Scope

Quant-Pulse-Processor is a local simulation library. It does not open network
connections, execute user-supplied code, deserialize untrusted data, or write
files outside the paths explicitly passed to `quantpulse bench --write`.

Relevant security concerns would be:

- denial of service through unbounded qubit counts or shot counts in a service
  that wraps the library,
- path traversal in the CLI's `--write` argument when the value comes from an
  untrusted source,
- supply-chain issues in the development dependencies (`pytest`, `ruff`).

The simulator validates qubit indices and matrix shapes, but it does not impose
a hard qubit cap: allocating `2^n` complex amplitudes is the caller's
responsibility. If you expose this library as a service, bound `n_qubits` and
`shots` at the boundary.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting:

1. Go to the [Security tab](https://github.com/MrKyaw/Quant-Pulse-Processor/security).
2. Select **Report a vulnerability**.
3. Include a description, a minimal reproduction, and the output of
   `quantpulse info`.

Do not open a public issue for a suspected vulnerability. You can expect an
acknowledgement within a few days; this is a personal project, so timelines are
best-effort rather than contractual.

## Dependency policy

The runtime dependency is NumPy only. Development dependencies are pinned by
lower bounds in `pyproject.toml` and installed fresh in CI on Python 3.10,
3.11 and 3.12.
