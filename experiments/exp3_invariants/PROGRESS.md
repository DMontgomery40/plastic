# exp3 progress log (for restartability)
- [x] branch experiments/harness-invariants created from dc55927
- [x] hypothesis installed in .venv
- [x] test_invariants.py written (step a draft)
- [ ] calib_records.py running for shipped+statements (logs experiments/logs/exp3_calib_*.log); pytest full run running (experiments/logs/exp3_pytest.log)
- [ ] fpr.py runs launched: shipped/shipped seeds 0-2 (--sessions 30)
- [x] full pytest (200 examples tiny, 8 real) done: 10 pass, 1 xfail(expected), 2 FAIL (findings):
      I1_strict (chunk rule, x=[0], flush -> pending chunk count/alpha_sum advanced by frozen read)
      I5 scale (delta rule, prefix [0]*8, chunk [0], scale 0.5 -> accepted 0.0247 > proposal 0.00621; decay not scaled)
      log copied to experiments/results/exp3/pytest_full.log
- [x] calib_records.py, fpr.py, analyze.py written
- [x] scale_real.py (forced scale on real model) - rerun with fixed summary
- [ ] sweep launched (18 fpr jobs via xargs -P7); shipped/shipped seeds 0-2 DONE; calib records DONE (0.01 recompute == shipped thresholds exactly)
- [x] analyze.py extended with reason_any / cusum alarm rate
- [x] WIP commit made on experiments/harness-invariants
- [x] all 21 fpr runs done; analyze.py run
- [x] README.md written
- [x] final commit (see git log)
