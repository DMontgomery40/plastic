"""One-off extra stats for report/exp2_tables.md (reads experiments/results/exp2/probe_seed*.json; writes nothing)."""
import json, glob, statistics as st
rows = [r for f in sorted(glob.glob('experiments/results/exp2/probe_seed*.json')) for r in json.load(open(f))]
for tag in ('direct', 'frozen'):
    rz = [r[tag + '_kl_rb_zero'] / r[tag + '_kl_rollback'] for r in rows]
    print(tag, 'n', len(rows), 'mean KL rollback %.4f rb_zero %.4f rb_rewind max %g commit %.4f' % (
        st.mean(r[tag + '_kl_rollback'] for r in rows), st.mean(r[tag + '_kl_rb_zero'] for r in rows),
        max(r[tag + '_kl_rb_rewind'] for r in rows), st.mean(r[tag + '_kl_commit'] for r in rows)))
    print('  rb_zero/rollback mean %.3f median %.3f; frac rb_zero<rollback %.3f' % (st.mean(rz), st.median(rz), sum(x < 1 for x in rz) / len(rz)))
    for b in ('0_16', '16_64', '64_256'):
        print('  bucket', b, 'rollback %.4f rb_zero %.4f commit %.4f' % tuple(st.mean(r[f'{tag}_kl_{c}_{b}'] for r in rows) for c in ('rollback', 'rb_zero', 'commit')))
    lk = [r[tag + '_kl_rollback'] / r[tag + '_kl_commit'] for r in rows]
    print('  overall leakage mean %.3f median %.3f min %.3f max %.3f' % (st.mean(lk), st.median(lk), min(lk), max(lk)))
