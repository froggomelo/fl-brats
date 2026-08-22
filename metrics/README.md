# Experiment metrics

Runtime metrics are written inside each run directory, not into this source folder.
Every completed federated run contains:

- `summary.json`: ET/TC/WT and mean Dice, HD95, sensitivity, best round, both
  convergence rounds, per-client results, and wall-clock time;
- `metrics/evaluation_rounds.jsonl`: global metrics by round;
- `metrics/clients.jsonl`: per-client Dice and supporting metrics;
- `communication/rounds.jsonl`: upload/download bytes and timing fields;
- `communication/summary.json`: cumulative bytes/MB and timing totals.

`network_transmission_seconds` is only populated by a genuine external collector.
The default `communication_and_coordination_seconds` is a low-overhead proxy and
is deliberately labelled as including serialization, RPC, scheduler waiting, and
aggregation overhead.
