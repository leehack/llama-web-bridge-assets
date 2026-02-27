# llama-web-bridge-assets

Versioned JavaScript/WASM artifacts for `leehack/llama-web-bridge`.

Primary files:

- `llama_webgpu_bridge.js`
- `llama_webgpu_bridge_worker.js`
- `llama_webgpu_core.js`
- `llama_webgpu_core.wasm`

Optional memory64 files (published in modern tags):

- `llama_webgpu_core_mem64.js`
- `llama_webgpu_core_mem64.wasm`

Metadata files:

- `manifest.json`
- `sha256sums.txt`

These files are intended for CDN consumption (for example via jsDelivr):

`https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge.js`

## Maintainer Docs

- `AGENTS.md`: artifact-repo workflow guidance
- `CONTRIBUTING.md`: publishing and update flow
