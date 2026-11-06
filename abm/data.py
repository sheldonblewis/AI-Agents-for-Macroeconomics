"""FRED series -> one quarterly frame of the macro drivers the agents react to."""
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SERIES = {
    'PCE': 'nominal personal consumption expenditures',
    'PCEPI': 'pce price index',
    'DSPIC96': 'real disposable personal income',
    'UNRATE': 'unemployment rate',
    'FEDFUNDS': 'effective federal funds rate',
    'CPIAUCSL': 'cpi, all urban consumers',
    'UMCSENT': 'university of michigan consumer sentiment',
    'PSAVERT': 'personal saving rate',
}
TRAIN = ('1980-01-01', '2009-12-31')
TEST = ('2010-01-01', '2030-01-01')
COVID = ('2020-01-01', '2021-06-30')


def download():
    (ROOT / 'data').mkdir(exist_ok=True)
    for sid in SERIES:
        urlretrieve(f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}', ROOT / 'data' / f'{sid}.csv')


def _series(sid):
    s = pd.read_csv(ROOT / 'data' / f'{sid}.csv', index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(s, errors='coerce')


def load():
    real_pce = _series('PCE') / _series('PCEPI') * 100
    m = pd.DataFrame({
        'pce': real_pce,
        'inc': _series('DSPIC96'),
        'u': _series('UNRATE'),
        'ff': _series('FEDFUNDS'),
        'cpi': _series('CPIAUCSL'),
        'sent': _series('UMCSENT'),
        'sav': _series('PSAVERT'),
    })
    q = m.resample('QS').mean()
    q['g_c'] = np.log(q.pce).diff() * 100        # real consumption growth, % q/q
    q['g_y'] = np.log(q.inc).diff() * 100        # real disposable income growth, % q/q
    q['infl'] = np.log(q.cpi).diff() * 400       # annualised inflation, %
    q['rr'] = q.ff - q.infl.rolling(4).mean()    # ex-post real policy rate, %
    q = q.loc['1979-01-01':].dropna()
    return q


def mask(index, span):
    return (index >= span[0]) & (index <= span[1])
