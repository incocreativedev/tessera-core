# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a vulnerability

If you discover a security vulnerability in Tessera, please report it responsibly. **Do not open a public issue.**

Email **kirk@incocreative.com** with:

- A description of the vulnerability
- Steps to reproduce
- Affected version(s)
- Impact assessment (what an attacker could achieve)

You will receive an acknowledgement within 48 hours. We aim to provide a fix or mitigation within 7 days for critical issues.

## Scope

The following areas are in scope for security reports:

- **Token signing** — Ed25519 signature bypass, key leakage, or verification flaws
- **Differential privacy** — Noise calibration errors that weaken (ε, δ) guarantees
- **TBF binary format** — Buffer overflows, CRC/HMAC bypass, or deserialisation attacks
- **Swarm validation** — Acceptance of malformed or malicious tokens
- **Audit integrity** — Tampering with or bypassing the audit log
- **Dependency vulnerabilities** — Issues in PyTorch, cryptography, safetensors, or msgpack that affect Tessera

## Out of scope

- Denial of service via large inputs (we don't claim resource bounds yet)
- Issues that require physical access to the machine running Tessera
- Vulnerabilities in dependencies that don't affect Tessera's usage of them

## Disclosure

We follow coordinated disclosure. Once a fix is released, we will credit the reporter (unless they prefer anonymity) in the changelog and release notes.

## PGP

If you need to encrypt your report, request our PGP key via email.
