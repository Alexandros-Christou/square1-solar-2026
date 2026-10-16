"""Local-only preparation. Use supplied competition CSVs, never external targets."""
import argparse
import hashlib
import json
from pathlib import Path
import os

# Bound OpenMP/BLAS work on the local Windows machine for predictable runtime.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def features(frame):
    time = pd.to_datetime(frame['datetime_utc'], utc=True, errors='raise')
    x = frame.drop(columns=['id', 'datetime_utc', 'generation_mw'], errors='ignore').apply(pd.to_numeric, errors='raise')
    x = x.replace(-999, np.nan)
    hour = time.dt.hour + time.dt.minute / 60
    day = time.dt.dayofyear
    for name, value, period in [('hour', hour, 24), ('day', day, 365.25)]:
        x[name + '_sin'] = np.sin(2 * np.pi * value / period)
        x[name + '_cos'] = np.cos(2 * np.pi * value / period)
    x['month'] = time.dt.month
    for suffix in ['ghi', 'clearsky', 'temp_c', 'cloud_pct', 'wind_ms', 'rh_pct']:
        cols = [c for c in frame if c.endswith('_' + suffix)]
        if cols:
            values = frame[cols].replace(-999, np.nan)
            x[suffix + '_mean'] = values.mean(axis=1)
            x[suffix + '_std'] = values.std(axis=1, ddof=0)
    return x


def run(train_path, test_path, output, synthetic=False):
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    for frame in [train, test]:
        required = {'id', 'datetime_utc', 'installed_capacity_mwp'}
        if not required.issubset(frame) or frame.id.isna().any() or frame.id.duplicated().any():
            raise ValueError('Missing required columns or invalid IDs')
        if not (frame.installed_capacity_mwp > 0).all():
            raise ValueError('Capacity must be positive')
    if 'generation_mw' in test or set(train.id) & set(test.id):
        raise ValueError('Unexpected test target or overlapping IDs')
    time = pd.to_datetime(train.datetime_utc, utc=True)
    test_time = pd.to_datetime(test.datetime_utc, utc=True)
    if time.max() >= test_time.min():
        raise ValueError('Training and test periods overlap')
    if not synthetic and (len(test) != 17520 or time.min() != pd.Timestamp('2022-01-01', tz='UTC') or test_time.min() != pd.Timestamp('2025-07-01', tz='UTC')):
        raise ValueError('Dataset does not match reviewed competition card')
    y = pd.to_numeric(train.generation_mw, errors='raise')
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Invalid training targets')
    split = pd.Timestamp('2024-07-01', tz='UTC')
    early, validation = time < split, time >= split
    if min(early.sum(), validation.sum()) < 20:
        raise ValueError('Insufficient chronological validation rows')
    x, xt = features(train), features(test)
    if set(x) != set(xt):
        raise ValueError('Train/test feature schema differs')
    xt = xt[x.columns]
    # Exclude features with no variation/information in the fitting period.
    # Select from training only, then apply the identical schema to test data.
    usable = x.loc[early].nunique(dropna=True) > 1
    x, xt = x.loc[:, usable], xt.loc[:, usable]
    capacity = train.installed_capacity_mwp
    normalised = y / capacity
    groups = pd.DataFrame({'month': time.dt.month, 'slot': time.dt.hour * 2 + time.dt.minute // 30, 'factor': normalised})
    lookup = groups.loc[early].groupby(['month', 'slot']).factor.mean()
    baseline = np.array([lookup.get((m, s), normalised[early].mean()) for m, s in zip(groups.loc[validation, 'month'], groups.loc[validation, 'slot'])]) * capacity[validation]
    results = []
    validation_predictions = {}
    for leaves in [15, 31]:
        model = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=160, learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=42)
        model.fit(x[early], normalised[early])
        prediction = np.maximum(0, model.predict(x[validation]) * capacity[validation])
        results.append((mean_absolute_error(y[validation], prediction), leaves))
        validation_predictions[leaves] = prediction
    score, leaves = min(results)
    final = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=160, learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=42)
    final.fit(x, normalised)
    prediction = np.maximum(0, final.predict(xt) * test.installed_capacity_mwp)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite predictions')
    output.mkdir(parents=True, exist_ok=True)
    errors = pd.DataFrame({'datetime_utc': time[validation], 'actual_mw': y[validation], 'prediction_mw': validation_predictions[leaves]})
    errors['absolute_error_mw'] = abs(errors.actual_mw - errors.prediction_mw)
    errors.to_csv(output / 'validation_predictions.csv', index=False)
    error_groups = {'month': errors.groupby(errors.datetime_utc.dt.month).absolute_error_mw.mean().to_dict(), 'hour_utc': errors.groupby(errors.datetime_utc.dt.hour).absolute_error_mw.mean().to_dict()}
    (output / 'error_analysis.json').write_text(json.dumps(error_groups, indent=2), encoding='utf-8')
    pd.DataFrame({'id': test.id, 'generation_mw': prediction}).to_csv(output / 'predictions.csv', index=False)
    joblib.dump(final, output / 'model.joblib')
    metrics = {'status': 'synthetic smoke test only' if synthetic else 'local chronological validation, not official leaderboard', 'validation_start': str(split), 'validation_rows': int(validation.sum()), 'baseline_mae_mw': mean_absolute_error(y[validation], baseline), 'models': [{'leaves': l, 'mae_mw': s} for s, l in results], 'selected_leaves': leaves, 'selected_mae_mw': score, 'test_rows': len(test), 'external_data_used': False, 'features': list(x), 'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [train_path, test_path]}}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in metrics.items() if k not in ['features', 'input_sha256']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train', type=Path)
    parser.add_argument('test', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    run(args.train, args.test, args.output, args.synthetic)
