import csv
import json
import os

RESULTS = "results"


def _dir(exp_id):
    d = os.path.join(RESULTS, exp_id)
    os.makedirs(d, exist_ok=True)
    return d


def save_config(exp_id, config):
    with open(os.path.join(_dir(exp_id), "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def save_metrics(exp_id, rows, fieldnames=None):
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(_dir(exp_id), "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def save_summary(exp_id, summary):
    with open(os.path.join(_dir(exp_id), "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
