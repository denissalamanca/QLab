# AFML Control Plane — Web (Phase 9)

React + Vite + TypeScript + Tailwind + shadcn/ui frontend for the CEO
Human-in-the-Loop control plane (Blueprint §11). Talks to the FastAPI backend
in `apps/api` (`afml.control_plane`).

## Features

- **Strategy dashboard** — CPCV-validated strategies awaiting sign-off
  (`GET /api/v1/registry/strategies`), with PBO / DSR ship-gate charts (Recharts).
- **Cryptographic approval modal** — the CEO signs the canonical message
  `afml:approve:<experiment_id>` on their own device (the private key never
  enters the browser), pastes the Ed25519 signature, and enters the live TOTP
  code. `POST /api/v1/execution/approve` verifies both before promoting
  Paper → Live.
- **Emergency flatten** — signature-only kill-switch over `afml:flatten:<nonce>`
  (`POST /api/v1/emergency/flatten`).

## Develop

```bash
npm install
npm run dev          # http://localhost:5173 (proxies /api → http://localhost:8000)
```

Run the backend alongside it:

```bash
# from the repo root, with AFML_CP_CEO_PUBLIC_KEY_HEX set + TOTP secret in the Keychain
uvicorn apps.api.main:app --port 8000
```

## End-to-end tests (Playwright)

```bash
npm run e2e:install   # one-time: install the Chromium browser
npm run e2e           # boots the dev server and runs tests/e2e
```

The approval spec intercepts the control-plane API so it runs deterministically
without a live backend or a real private key, while still asserting the §11.1
request contract.

## Build

```bash
npm run build         # tsc -b && vite build → dist/
```

> Note: this app is **not** part of the Python `make phase9` gate (no JS in the
> CI Python toolchain). The gated, security-critical surface is the backend +
> crypto under `src/afml`. Run `npm run typecheck` / `npm run e2e` in a Node
> environment to validate the frontend.
