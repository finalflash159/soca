# Retrieval benchmark data

`sources.lock.json` is the tracked source of truth. Raw and prepared payloads
are local artifacts and remain ignored because they include large or
redistribution-constrained datasets.

Provision all pinned sources:

```bash
uv run python scripts/provision_retrieval_eval_data.py --all
```

The provisioner writes `provisioned-manifest.json` with byte sizes and SHA-256
digests for every selected file. Quality runners must reject `demo_smoke` and
`unit_fixture`; the user demo vault and derivatives are never valid benchmark
inputs.

`private/` is always local-only. Private corpus text, paths, qrels and queries
must not be committed or sent to a remote evaluator. Only aggregate metrics and
an opaque manifest hash may leave the machine.
