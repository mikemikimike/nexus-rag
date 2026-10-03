# Development

Python 3.12. The package is in `src/local_rag`.

```bash
pip install -e . --group dev   # or: uv pip install -e . --group dev
pytest
```

Tests use an in-memory Corpus, a fake Model gateway and in-process Qdrant, plus fake `lit`, `confluence`, `docker` and host commands, so they need no external service. `local_rag.smoke` and `local_rag.acceptance` check the real ones (the acceptance suite's own logic is tested against the fakes, with a deliberately broken fake for each failure path).

| Module | Role |
|---|---|
| `index`, `generations`, `index_format`, `exclusive`, `journal`, `ranking`, `sparse`, `chunker`, `extraction`, `settings` | the Knowledge index and what sits behind it |
| `corpus`, `confluence`, `folder`, `drawio` | the Corpus seam and its two adapters |
| `gateway`, `gateway_litellm`, `fakes` | the Model gateway seam, its LiteLLM adapter and the deterministic fake |
| `cli`, `mcp_server`, `environment` | adapters: `local-rag`, `local-rag-mcp`, and settings from environment variables |
| `installer`, `container`, `binary_drop` | delivery: the Installer, the container route and Binary drop verification |
| `smoke` | the manual smoke check |
| `checks` | what the smoke check and the acceptance suite share: how a check reports, and the embedding and rerank checks |
| `acceptance`, `acceptance_report`, `acceptance_corpus`, `acceptance_stack`, `acceptance_index`, `acceptance_mcp`, `acceptance_host` | the live acceptance suite: its entry point, what its check groups work from, its generated Documents, and one module per check group |

`deploy/systemd/` and `deploy/container/` hold the unit files, the compose file and the Dockerfile; `examples/` holds the client and proxy examples. To regenerate `requirements.lock` after a dependency change, follow the comment at its top.
