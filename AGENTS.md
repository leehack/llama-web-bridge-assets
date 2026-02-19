# AGENTS.md

Guidance for coding agents working in `llama-web-bridge-assets`.

## Scope and Ownership

- This repository stores versioned published bridge assets.
- Primary producer is `llama-web-bridge` publish workflow.
- Consumers (for example `llamadart`) pin tags from this repo.

## Expected Contents

- `llama_webgpu_bridge.js`
- `llama_webgpu_core.js`
- `llama_webgpu_core.wasm`
- `manifest.json`
- `sha256sums.txt`

## Publishing Model

Preferred: publish from `llama-web-bridge` workflow
`.github/workflows/publish_assets.yml`.

That workflow commits updated assets and creates a new tag in this repo.

## Change Boundaries

- Avoid manual edits to built JS/WASM files.
- Manual changes should usually be limited to repository documentation.
- Runtime behavior changes should be made in `llama-web-bridge`, then republished.
