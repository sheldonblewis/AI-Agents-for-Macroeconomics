"""Calibrate the agent economy on 1980-2009 and forecast real consumption growth 2010-2026.

    python run.py      # ~2-4 min, rewrites results/ and figures/
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import differential_evolution  # noqa: E402

from abm import data  # noqa: E402
from abm.model import ARCHETYPES, BOUNDS, PARAMS, Economy, State, mpc, simulate  # noqa: E402

DRIVERS = ['g_y', 'u', 'rr', 'sent']
OUT, FIG = Path('results'), Path('figures')
COLORS = ['#7d93b8', '#c9a46c', '#8fb39a', '#b88a9f']
plt.rcParams.update({'figure.dpi': 140, 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.25, 'legend.frameon': False})

q = data.load()
train = data.mask(q.index, data.TRAIN)
test = data.mask(q.index, data.TEST)
covid = data.mask(q.index, data.COVID)
y = q.g_c.to_numpy()
target = np.append(y[1:], np.nan)           # the quarter being forecast from t
fit_rows = train & ~np.isnan(target)

# driver forecasts: an ar(1) per driver, fit on the training sample only
X = q[DRIVERS].to_numpy()
ar = {}
for j, name in enumerate(DRIVERS):
    x0, x1 = X[:-1][train[:-1]], X[1:][train[:-1]]
    b, a = np.polyfit(x0[:, j], x1[:, j], 1)
    ar[name] = (a, b)
X_fc = np.column_stack([ar[n][0] + ar[n][1] * X[:, j] for j, n in enumerate(DRIVERS)])

econ = Economy(n=20000, horizon=len(q) + 2, seed=0)


def raw_forecasts(theta):
    return simulate(econ, [tuple(r) for r in X], [tuple(r) for r in X_fc], theta)


def mapped(fc, a_b=None):
    """Bias correction from model growth to measured pce growth, fit in-sample.

    Slope is fixed at 1: an earlier free affine map (results/table_v2_inverted.md) fit a
    negative slope, i.e. it scored well by inverting the model, which is not a forecast.
    """
    if a_b is None:
        a_b = np.array([1.0, np.mean(target[fit_rows] - fc[fit_rows])])
    return np.polyval(a_b, fc), a_b


def objective(theta):
    _, fc, _, _ = raw_forecasts(theta)
    if not np.all(np.isfinite(fc[fit_rows])):
        return 1e6
    pred, _ = mapped(fc)
    return np.mean((pred[fit_rows] - target[fit_rows]) ** 2)


def ols_forecast(features):
    Z = np.column_stack([np.ones(len(q)), features])
    beta, *_ = np.linalg.lstsq(Z[fit_rows], target[fit_rows], rcond=None)
    return Z @ beta


def dm_test(e1, e2):
    """Diebold-Mariano on squared errors, newey-west with one lag. Negative = model 1 better."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    g0 = np.var(d)
    g1 = np.mean((d[1:] - d.mean()) * (d[:-1] - d.mean()))
    se = np.sqrt((g0 + 2 * 0.5 * g1) / n)
    t = d.mean() / se
    return t, 2 * (1 - stats.norm.cdf(abs(t)))


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    res = differential_evolution(objective, BOUNDS, maxiter=30, popsize=10, seed=0, tol=1e-6,
                                 polish=True, updating='deferred', workers=-1)
    theta = res.x
    print('calibrated:', dict(zip(PARAMS, np.round(theta, 3))), 'train mse', round(res.fun, 4))
    C, fc, by_k, fc_k = raw_forecasts(theta)
    abm, a_b = mapped(fc)

    lag_feats = q[['g_c'] + DRIVERS].to_numpy()
    forecasts = {
        'agent-based model': abm,
        'historical mean': np.full(len(q), target[fit_rows].mean()),
        'random walk': y.copy(),
        'ar(2)': ols_forecast(np.column_stack([y, np.append(np.nan, y[:-1])])),
        'linear on same drivers (arx)': ols_forecast(lag_feats),
    }
    rows = test & ~np.isnan(target)
    rows_ex = rows & ~covid & ~np.append(covid[1:], False)

    table = {}
    for name, f in forecasts.items():
        e = f - target
        table[name] = {
            'rmse': float(np.sqrt(np.mean(e[rows] ** 2))),
            'rmse_ex_covid': float(np.sqrt(np.mean(e[rows_ex] ** 2))),
            'mae_ex_covid': float(np.mean(np.abs(e[rows_ex]))),
            'direction_ex_covid': float(np.mean(np.sign(f[rows_ex] - y[rows_ex]) == np.sign(target[rows_ex] - y[rows_ex]))),
        }
    for name in forecasts:
        if name == 'agent-based model':
            continue
        t, p = dm_test((abm - target)[rows_ex], (forecasts[name] - target)[rows_ex])
        table[name]['dm_vs_abm_ex_covid'] = {'t': float(t), 'p': float(p)}

    # mpcs in 2019q4 and impulse responses from that state
    t0 = int(np.where(q.index == '2019-10-01')[0][0])
    s = filter_to(theta, t0)
    d0 = tuple(X[t0])
    m = mpc(econ, s, d0, theta, t0 + 1)
    mpc_by = {a: float(np.mean(m[econ.k == i])) for i, a in enumerate(ARCHETYPES)}
    irfs = impulse(theta, s, d0, t0)

    summary = {'params': dict(zip(PARAMS, map(float, theta))), 'train_mse': float(res.fun),
               'measurement': {'intercept': float(a_b[1]), 'slope': float(a_b[0])},
               'driver_ar1': ar, 'forecast_table': table, 'mpc_2019q4': mpc_by,
               'test_quarters': int(rows.sum()), 'test_quarters_ex_covid': int(rows_ex.sum())}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    write_table(table, mpc_by, theta)

    plots(forecasts, rows, abm, fc_k, a_b, m, irfs)


def filter_to(theta, t_end):
    s = econ.init()
    for _ in range(12):
        s, _ = econ.step(s, (0.5, X[0][1], X[0][2], X[0][3]), theta, 0)
    for t in range(t_end + 1):
        s, _ = econ.step(s, tuple(X[t]), theta, t)
    return s


def impulse(theta, s0, d0, t0, H=10):
    """Consumption response by archetype to one-off driver shocks that decay at 0.75 per quarter."""
    shocks = {'+1pp real rate': (2, 1.0), '+2pp unemployment': (1, 2.0), '-2% real income': (0, -2.0)}
    base_path = run_path(theta, s0, [d0] * H, t0)
    out = {}
    for name, (j, size) in shocks.items():
        path = []
        for h in range(H):
            d = list(d0)
            # income is a growth rate, so its shock is a one-quarter level drop; the rest decay
            d[j] += (size if h == 0 else 0.0) if j == 0 else size * 0.75 ** h
            path.append(tuple(d))
        shocked = run_path(theta, s0, path, t0)
        out[name] = (shocked / base_path - 1) * 100
    return out


def run_path(theta, s, path, t0):
    s = State(s.perm.copy(), s.wealth.copy(), s.employed.copy(), s.belief)
    by_k = []
    for h, d in enumerate(path):
        s, c = econ.step(s, d, theta, t0 + 1 + h)
        by_k.append(np.bincount(econ.k, c, 4))
    return np.array(by_k)


def write_table(table, mpc_by, theta):
    lines = ['| forecaster | rmse (all) | rmse ex-covid | mae ex-covid | direction ex-covid | dm t vs abm (p) |',
             '|---|---|---|---|---|---|']
    for name, v in sorted(table.items(), key=lambda kv: kv[1]['rmse_ex_covid']):
        dm = v.get('dm_vs_abm_ex_covid')
        dm_s = f"{-dm['t']:.2f} ({dm['p']:.2f})" if dm else '-'
        lines.append(f"| {name} | {v['rmse']:.2f} | {v['rmse_ex_covid']:.2f} | {v['mae_ex_covid']:.2f} | "
                     f"{v['direction_ex_covid']:.0%} | {dm_s} |")
    lines += ['', '| archetype | mpc out of a one-off transfer (2019q4) |', '|---|---|']
    lines += [f'| {a} | {v:.2f} |' for a, v in mpc_by.items()]
    lines += ['', 'calibrated parameters: ' + ', '.join(f'{p}={v:.3f}' for p, v in zip(PARAMS, theta))]
    (OUT / 'table.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def plots(forecasts, rows, abm, fc_k, a_b, m, irfs):
    idx = q.index
    nxt = idx + np.timedelta64(92, 'D')
    r = np.where(rows)[0]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(nxt[r], target[r], color='k', lw=1.3, label='actual real pce growth')
    ax.plot(nxt[r], forecasts['linear on same drivers (arx)'][r], color='#bbb', lw=1, label='linear (arx)')
    ax.plot(nxt[r], abm[r], color='#1f3a5f', lw=1.6, label='agent-based model')
    ax.set_ylim(-3, 4)
    ax.annotate('covid quarters off scale\n(-9% and +8%)', (np.datetime64('2018-03-01'), 3.3), fontsize=7, ha='center', color='#666')
    ax.set_ylabel('% q/q')
    ax.set_title('one-quarter-ahead forecasts of real consumption growth, out of sample')
    ax.legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'forecast_test.png')
    plt.close(fig)

    # contributions by archetype (model units, mapped with the same slope)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    contrib = fc_k[r] * a_b[0]
    bottom_pos = np.zeros(len(r))
    bottom_neg = np.zeros(len(r))
    for i, a in enumerate(ARCHETYPES):
        v = contrib[:, i]
        ax.bar(nxt[r], np.where(v > 0, v, 0), 60, bottom=bottom_pos, color=COLORS[i], label=a)
        ax.bar(nxt[r], np.where(v < 0, v, 0), 60, bottom=bottom_neg, color=COLORS[i])
        bottom_pos += np.where(v > 0, v, 0)
        bottom_neg += np.where(v < 0, v, 0)
    ax.set_ylim(-3, 3)
    ax.set_title('who drives forecast consumption growth: contribution by archetype')
    ax.set_ylabel('pp of growth')
    ax.legend(fontsize=7, ncol=4, loc='lower left')
    fig.tight_layout()
    fig.savefig(FIG / 'archetype_contributions.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=False)
    for ax, (name, resp) in zip(axes, irfs.items()):
        for i, a in enumerate(ARCHETYPES):
            ax.plot(range(1, len(resp) + 1), resp[:, i], color=COLORS[i], lw=1.5, label=a)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(name)
        ax.set_xlabel('quarters after shock')
    axes[0].set_ylabel('% change in consumption')
    axes[0].legend(fontsize=6)
    fig.suptitle('impulse responses by archetype, from the 2019q4 state', fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / 'impulse_responses.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.linspace(0, 1, 26)
    for i, a in enumerate(ARCHETYPES):
        ax.hist(np.clip(m[econ.k == i], 0, 1), bins, color=COLORS[i], alpha=0.65, label=a, density=True)
    ax.set_xlabel('mpc out of a one-off transfer')
    ax.set_title('heterogeneous mpcs: constrained households spend it all')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / 'mpc_distribution.png')
    plt.close(fig)

    t = json.loads((OUT / 'summary.json').read_text())['forecast_table']
    names = sorted(t, key=lambda k: t[k]['rmse_ex_covid'])
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    yy = np.arange(len(names))
    ax.barh(yy - 0.18, [t[k]['rmse_ex_covid'] for k in names], 0.36, color=['#1f3a5f' if k == 'agent-based model' else '#bbb' for k in names], label='ex-covid')
    ax.set_yticks(yy, names)
    ax.invert_yaxis()
    ax.set_xlabel('out-of-sample rmse, pp (2010-2026 excluding 2020q1-2021q2)')
    fig.tight_layout()
    fig.savefig(FIG / 'rmse_test.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
