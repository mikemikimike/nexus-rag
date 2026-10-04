# Development

Python 3.12. The package is in `src/nexus_rag`.

```bash
pip install -e . --group dev   # or: uv pip install -e . --group dev
pytest
```

Tests use an in-memory Corpus, a fake Model gateway and in-process Qdrant, plus fake `lit`, `confluence`, `docker` and host commands, so they need no external service. `nexus_rag.smoke` and `nexus_rag.acceptance` check the real ones (the acceptance suite's own logic is tested against the fakes, with a deliberately broken fake for each failure path).

| Module | Role |
|---|---|
| `index`, `generations`, `index_format`, `index_names`, `passage_store`, `exclusive`, `journal`, `ranking`, `sparse`, `settings` | the Knowledge index and what sits behind it |
| `chunker`, `tokens`, `syntax`, `languages`, `extraction`, `drawio` | turning a Document into Passages: text extraction, token counting, and code split along its syntax |
| `corpus`, `declaration`, `confluence`, `folder`, `gitlab` | the Corpus seam, the Corpus declaration file, and its three adapters |
| `gateway`, `gateway_litellm`, `fakes` | the Model gateway seam, its LiteLLM adapter and the deterministic fake |
| `cli`, `__main__`, `mcp_server`, `environment` | adapters: `nexus-rag` (also `python -m nexus_rag`), `nexus-rag-mcp`, and settings from environment variables |
| `installer`, `container`, `binary_drop` | delivery: the Installer, the container route and Binary drop verification |
| `smoke` | the manual smoke check |
| `checks` | what the smoke check and the acceptance suite share: how a check reports, and the embedding and rerank checks |
| `errors` | how an error reads in a report |
| `acceptance`, `acceptance_report`, `acceptance_corpus`, `acceptance_stack`, `acceptance_index`, `acceptance_mcp`, `acceptance_host` | the live acceptance suite: its entry point, what its check groups work from, its generated Documents, and one module per check group |

`deploy/systemd/` and `deploy/container/` hold the unit files, the compose file and the Dockerfile; `examples/` holds the client and proxy examples. To regenerate `requirements.lock` after a dependency change, follow the comment at its top.
