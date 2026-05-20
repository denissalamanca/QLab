"""``afml`` operator CLI (Operational Milestone M0).

Entry point registered under ``[project.scripts]`` so the operator runs
``afml enroll-ceo`` / ``afml doctor`` directly. Thin by design: all logic lives
in :mod:`afml.crypto` and :mod:`afml.ops`.
"""

from __future__ import annotations

from typing import Annotated

import typer

from afml.crypto import (
    CEO_PRIVATE_KEY,
    CEO_TOTP_SECRET,
    SecretNotFoundError,
    delete_secret,
    generate_keypair,
    get_or_create_ceo_totp_secret,
    load_secret,
    public_key_from_private,
    store_ceo_private_key,
)
from afml.ops import run_health_checks

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="AFML Quant Lab — operator CLI.",
)


@app.command("enroll-ceo")
def enroll_ceo(
    force: Annotated[
        bool, typer.Option("--force", help="Rotate keys even if already enrolled.")
    ] = False,
) -> None:
    """Enrol the CEO: generate the Ed25519 keypair + TOTP seed, persist to Keychain.

    Prints the public-key hex (for ``AFML_CP_CEO_PUBLIC_KEY_HEX``) and the
    ``otpauth://`` provisioning URI **once**. The private key is stored in the OS
    Keychain and never leaves this device.
    """
    try:
        existing = load_secret(CEO_PRIVATE_KEY)
    except SecretNotFoundError:
        existing = None

    if existing is not None and not force:
        pub = public_key_from_private(bytes.fromhex(existing))
        typer.echo("CEO already enrolled (use --force to rotate keys).")
        typer.echo(f"CEO public key (AFML_CP_CEO_PUBLIC_KEY_HEX): {pub.hex()}")
        raise typer.Exit(code=0)

    if force:
        delete_secret(CEO_PRIVATE_KEY)
        delete_secret(CEO_TOTP_SECRET)

    private_key, public_key = generate_keypair()
    store_ceo_private_key(private_key.hex())
    # Persists the seed + echoes the otpauth:// URI once (Ops M0 / audit P3).
    get_or_create_ceo_totp_secret(echo=typer.echo)

    typer.echo("")
    typer.echo(f"CEO public key (set AFML_CP_CEO_PUBLIC_KEY_HEX): {public_key.hex()}")
    typer.echo("Private key stored in the OS Keychain — it never leaves this device.")


@app.command()
def doctor() -> None:
    """Health-check the lab: data, warm-up history, Redis, registry, Keychain.

    Exits 0 when every check passes, 1 otherwise (with a per-check report).
    """
    report = run_health_checks()
    for check in report.checks:
        mark = "PASS" if check.ok else "FAIL"
        typer.echo(f"[{mark}] {check.name}: {check.detail}")
    if not report.healthy:
        typer.echo(f"\n{len(report.failures())} check(s) failed.")
        raise typer.Exit(code=1)
    typer.echo("\nAll systems healthy.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
