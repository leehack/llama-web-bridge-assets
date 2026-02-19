# Contributing to llama-web-bridge-assets

This repository is a versioned artifact store.

## Purpose

It contains published bridge runtime files used by downstream consumers via
pinned tags and CDN URLs.

## Recommended Update Flow

1. Make runtime changes in `llama-web-bridge`.
2. Run `Publish Bridge Assets` workflow there.
3. Workflow updates files in this repo and creates release tag.
4. Update consuming repos (for example `llamadart`) to the new tag.

## Manual Changes

Manual edits to generated JS/WASM files are discouraged.
If unavoidable, regenerate checksums and keep `manifest.json` accurate.

## Consumer Notes

A typical consumer URL pattern:

`https://cdn.jsdelivr.net/gh/leehack/llama-web-bridge-assets@<tag>/llama_webgpu_bridge.js`
