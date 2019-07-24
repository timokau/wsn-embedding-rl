"""Figure out good terminating conditions from experience
One-off script, not written for readability or maintainability.
Intended to come up with a dynamic stop condition for hyperparameter
search, though that will probably not actually happen due to time
constraints.
"""

from collections import defaultdict
import os
import csv
import numpy as np

STEPSIZE = 2000


def _evaluate_csv(csvfile):
    # pylint: disable=too-many-locals
    result = defaultdict(list)
    with open(csvfile, "r") as f:
        reader = csv.reader(f)
        keys = next(reader)
        for line in reader:
            for key, val in zip(keys, line):
                result[key].append(val)
    rewards = [int(float(val)) for val in result["reward"] if val != ""]
    steps = [int(float(val)) for val in result["steps"] if val != ""]
    assert len(rewards) == len(steps)
    cur = 0
    reward_list = []
    averages = []
    variances = []
    episodes = []
    for (i, (s, r)) in enumerate(zip(steps, rewards)):
        if s // STEPSIZE > cur:
            averages.append(np.average(reward_list))
            variances.append(np.var(reward_list))
            episodes.append(i)

            cur = s // STEPSIZE
            reward_list = []

        reward_list.append(r)

    last_avg = -np.infty
    last_var = np.infty
    for avg, var, episode in zip(averages, variances, episodes):
        avg_improvement = avg - last_avg
        var_improvement = var - last_var
        last_avg = avg
        last_var = var
        would_abort = avg_improvement < 0 and var_improvement < 0
        if would_abort:
            hindsight = avg - averages[-1]
            return (
                steps[episode],
                steps[-1],
                episode,
                avg_improvement,
                hindsight,
            )

    return (None, steps[-1], len(rewards), avg_improvement, None)


def _main():
    basedir = "/home/timo/vm/wsn-embedding-rl/logs/"
    result = 0
    hindsights = []
    for experiment_dir in os.listdir(basedir):
        (abort_at, total, eps, imp, hindsight) = _evaluate_csv(
            os.path.join(basedir, experiment_dir, "progress.csv")
        )
        if abort_at is not None:
            result -= total - abort_at
            hindsights.append(hindsight)
            print(
                experiment_dir,
                abort_at,
                total,
                eps,
                round(imp, 2),
                round(hindsight, 2),
            )
        elif abort_at is None and total > 30000:
            result += 10000
            print(experiment_dir)
    print(result, min(hindsights), np.average(hindsights))


if __name__ == "__main__":
    _main()
