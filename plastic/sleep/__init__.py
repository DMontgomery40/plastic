"""Sleep: offline consolidation of session traces into the slow weights, gated by the canaries."""

# HuggingFaceTB/smoltalk main, checked 2026-09-23: the pinned dataset revision for replay and held-out rows. Lives here,
# torch-free, so the CLI parser and experiment scripts share one value without importing the sleep implementation.
SMOLTALK_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
