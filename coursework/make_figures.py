"""Rebuild the draft's figures from its frozen, checked experiment summaries."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'evidence/results.json').read_text())
training_status = data.get('training_status', {})
finished = training_status.get('stop_reason') == 'selfplay_game_limit'
training_timestamp = training_status['utc'] if finished else data['captured_utc']
capture_label = datetime.fromisoformat(training_timestamp).strftime('%d %b %Y, %H:%M UTC')
capture_label = f"{'Completed' if finished else 'Captured'} {capture_label}"
rows = data['training']
benchmarks = data['benchmarks']
details = json.loads((ROOT / 'evidence/benchmark-details.json').read_text())
assert len(rows) == data['evaluated_candidates']
assert [r['iteration'] for r in rows] == list(range(1, len(rows) + 1))
start_iteration = data['production_plan']['persistent_start_iteration']
rows = [r for r in rows if r['iteration'] >= start_iteration]
assert start_iteration == 183 and rows
assert all(r['mode'] == 'persistent_raw_v1' for r in rows)
assert [r['iteration'] - start_iteration for r in rows] == list(range(len(rows)))
benchmark_iterations = [b['iteration'] for b in benchmarks]
assert benchmark_iterations == sorted(set(benchmark_iterations))
assert set(benchmark_iterations) <= set(data['promotions'])
assert all(i >= start_iteration for i in benchmark_iterations)
for b in benchmarks:
    assert b['wins'] + b['draws'] + b['losses'] == b['games'] == 128
    assert np.isclose(b['score'], (b['wins'] + .5 * b['draws']) / b['games'])
    assert len(b['pair_scores']) == 64 and np.isclose(np.mean(b['pair_scores']), b['score'])
assert len({b['reference_sha256'] for b in benchmarks}) == 1
assert len({b['opening_seed'] for b in benchmarks}) == 1
best_benchmark_iteration = max(benchmarks, key=lambda b: b['score'])['iteration']
assert details['opening_seed'] == benchmarks[0]['opening_seed']
assert len(details['openings']) == 64
assert all(len(opening) in (0, 2, 4, 6, 8, 10) for opening in details['openings'])
assert [b['iteration'] for b in details['benchmarks']] == benchmark_iterations
rewards = np.array([[b['player1_rewards'], b['player2_rewards']]
                    for b in details['benchmarks']])
assert rewards.shape == (len(benchmarks), 2, 64)
assert np.isin(rewards, [-1, 0, 1]).all()
seat_scores = (rewards + 1) / 2
pair_scores = np.array([b['pair_scores'] for b in benchmarks])
assert np.array_equal(seat_scores.mean(axis=1), pair_scores)
for index, b in enumerate(benchmarks):
    for reward, key in ((1, 'wins'), (0, 'draws'), (-1, 'losses')):
        assert np.count_nonzero(rewards[index] == reward) == b[key]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titlesize': 9, 'axes.labelsize': 8,
                     'legend.fontsize': 6.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
teal, purple, grey = '#087f8c', '#7652a3', '#687787'
x = np.array([r['iteration'] - start_iteration for r in rows])
score = np.array([r['score'] for r in rows]) * 100
ci = np.array([r['ci95'] for r in rows]) * 100
promotions = [r for r in rows if r['promoted']]
assert [r['iteration'] for r in promotions] == [i for i in data['promotions'] if i >= start_iteration]
assert np.all(ci[:, 0] <= score) and np.all(score <= ci[:, 1])

def selfplay_axis(ax, title):
    mapping = data['persistent_selfplay_axis']
    origin = mapping['candidate_zero_games']
    stride = mapping['games_per_candidate']
    final_batch = mapping.get('final_batch_games', stride)
    last_full_candidate = x[-1] - 1
    games_before_final = origin + last_full_candidate * stride
    assert 0 < final_batch <= stride

    def to_games(candidate):
        adjustment = np.maximum(candidate - last_full_candidate, 0) * (final_batch - stride)
        return (origin + candidate * stride + adjustment) / 1000

    def to_candidate(thousands):
        games = thousands * 1000
        return np.where(games <= games_before_final, (games - origin) / stride,
                        last_full_candidate + (games - games_before_final) / final_batch)

    if 'final_batch_games' in mapping:
        assert to_games(x[-1]) == data['selfplay_games'] / 1000
        assert np.allclose(to_candidate(to_games(x[-2:])), x[-2:])
    top = ax.secondary_xaxis('top', functions=(to_games, to_candidate))
    top.xaxis.set_major_locator(MultipleLocator(100))
    top.set_xlabel('Cumulative self-play games (thousands)', labelpad=4)
    top.tick_params(labelsize=7)
    ax.set_title(title, pad=34)


def candidate_panel(ax, title):
    ax.errorbar(x, score, yerr=[score-ci[:, 0], ci[:, 1]-score], fmt='.',
                color=teal, ecolor='#c8dadd', markersize=2.1, elinewidth=.4, alpha=.8)
    ax.axhline(50, color=grey, linestyle='--', linewidth=.8)
    ax.axhline(55, color='#b38b3a', linestyle=':', linewidth=.9)
    boundaries = [-1] + [r['iteration'] - start_iteration for r in promotions] + [int(x[-1]) + 1]
    for start, end in zip(boundaries, boundaries[1:]):
        keep = (x > start) & (x <= end)
        if keep.sum() >= 10:
            ax.plot(x[keep][9:], np.convolve(score[keep], np.ones(10)/10, mode='valid'),
                    color='#253044', linewidth=1)
    for r in promotions:
        ax.scatter(r['iteration'] - start_iteration, 100*r['score'],
                   color='#27824f', marker='*', s=35, zorder=4)
    ax.set(xlabel='Candidate iteration (persistent / raw targets)',
           ylabel='Score (%)', ylim=(35, 66), xlim=(-2, int(x[-1]) + 4))
    ax.text(.03, .035, 'Dots: candidates; stars: promotions\nLine: 10-candidate mean within each opponent',
            transform=ax.transAxes, fontsize=6.4, color=grey)
    ax.grid(alpha=.18)
    selfplay_axis(ax, title)


fig, ax = plt.subplots(figsize=(7.5, 3.43))
candidate_panel(ax, 'Candidate vs retained champion')
ax.text(.99, .035, 'Candidate 0 = original log iteration 183\n'
        f'128 games/candidate; paired 95% intervals\n{capture_label}',
        ha='right', transform=ax.transAxes, fontsize=6.4, color=grey)
fig.tight_layout(pad=.8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/candidate-progress.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

# Joint opening resamples retain correlations between checkpoint scores.
bootstrap_indices = np.random.default_rng(20261118).integers(0, 64, (50000, 64))
benchmark_x = np.array(benchmark_iterations) - start_iteration
design = np.column_stack([np.ones(len(benchmarks)), benchmark_x / 100])
score_percent = pair_scores * 100
standard_errors = score_percent.std(axis=1, ddof=1) / np.sqrt(pair_scores.shape[1])
assert np.all(standard_errors > 0)
weights = 1 / standard_errors
weights /= weights.sum()
first, second = np.triu_indices(len(benchmarks), k=1)
x_scaled = design[:, 1]
dx = x_scaled[second] - x_scaled[first]


def fit_absolute_errors(scores):
    """An optimal weighted LAD line can be chosen through two data points."""
    coefficients = []
    for start in range(0, len(scores), 256):
        y = scores[start:start + 256]
        slopes = (y[:, second] - y[:, first]) / dx
        offsets = y[:, first] - slopes * x_scaled[first]
        residuals = y[:, None, :] - offsets[:, :, None] - slopes[:, :, None] * x_scaled
        losses = np.einsum('ijk,k->ij', np.abs(residuals), weights)
        winners = np.argmin(losses, axis=1)
        batch = np.arange(len(winners))
        coefficients.append(np.column_stack([offsets[batch, winners], slopes[batch, winners]]))
    return np.concatenate(coefficients)


fit_coefficients = fit_absolute_errors(score_percent.mean(axis=1)[None, :])[0]
bootstrap_scores = np.array([values[bootstrap_indices].mean(axis=1) for values in score_percent])
bootstrap_coefficients = fit_absolute_errors(bootstrap_scores.T)
fit_x = np.linspace(benchmark_x.min(), benchmark_x.max(), 200)
fit_design = np.column_stack([np.ones(len(fit_x)), fit_x / 100])
fit_y = fit_design @ fit_coefficients
fit_band = np.quantile(bootstrap_coefficients @ fit_design.T, [.025, .975], axis=0)
slope_interval = np.quantile(bootstrap_coefficients[:, 1], [.025, .975])
assert np.isfinite(fit_band).all() and np.all(fit_band[0] <= fit_band[1])
fit_record = {
    'method': 'Weighted least absolute deviations with inverse opening-pair standard-error weights',
    'candidates': benchmark_x.tolist(), 'intercept_percent': float(fit_coefficients[0]),
    'slope_pp_per_100_candidates': float(fit_coefficients[1]),
    'slope_ci95_pp_per_100_candidates': slope_interval.tolist(),
    'standard_errors_percent': standard_errors.tolist(),
    'normalized_weights': weights.tolist(),
    'weighted_mae_percent': float(np.abs(score_percent.mean(axis=1) - design @ fit_coefficients) @ weights),
    'bootstrap_seed': 20261118, 'bootstrap_resamples': 50000,
    'bootstrap_weights_held_fixed': True,
    'band': 'Pointwise 95% percentile interval for the fitted line; not a prediction band',
    'fit_range': [int(benchmark_x.min()), int(benchmark_x.max())],
    'limitations': 'Descriptive trend over selected checkpoints; conditional on estimated weights and reused openings, without selection correction or extrapolation.'}
(ROOT / 'evidence/benchmark-linear-fit.json').write_text(json.dumps(fit_record, indent=2) + '\n')

def benchmark_panel(ax, title):
    low_limit = min(40, min(b['paired_ci95'][0] * 100 for b in benchmarks) - 2)
    high_limit = max(60, max(b['paired_ci95'][1] * 100 for b in benchmarks) + 2)
    ax.fill_between(fit_x, fit_band[0], fit_band[1], color=teal, alpha=.13, linewidth=0,
                    label='95% fitted-line band')
    ax.plot(fit_x, fit_y, color=teal, linewidth=1.35, label='Weighted LAD fit')
    for b in benchmarks:
        point = 100*b['score']
        low, high = np.array(b['paired_ci95']) * 100
        assert low <= point <= high
        color = purple if b['iteration'] == best_benchmark_iteration else '#253044'
        ax.errorbar(b['iteration'] - start_iteration, point,
                    yerr=[[point-low], [high-point]], fmt='o', color=color,
                    capsize=2.5, markersize=3.5, linewidth=.9, zorder=3,
                    label='Highest score' if b['iteration'] == best_benchmark_iteration else None)
    ax.axhline(50, color=grey, linestyle='--', linewidth=.8)
    ax.set(xlabel='Promoted candidate iteration', ylabel='Score (%)',
           xlim=(0, benchmark_x.max() + 20),
           ylim=(min(low_limit, fit_band[0].min() - 1), max(high_limit, fit_band[1].max() + 1) + 1))
    ax.legend(loc='upper left', frameon=False, fontsize=6, handlelength=1.8)
    ax.text(.025, .025, f'Trend: {fit_coefficients[1]:+.2f} pp / 100 candidates',
            transform=ax.transAxes, fontsize=6.2, color=grey)
    ax.grid(alpha=.18)
    selfplay_axis(ax, title)


fig, ax = plt.subplots(figsize=(6.65, 3.53))
benchmark_panel(ax, 'Promoted candidates vs frozen original submission')
fig.tight_layout(pad=.8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/fixed-original-benchmarks.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(6.65, 3.43),
                         gridspec_kw={'width_ratios': [1.15, 1]})
candidate_panel(axes[0], '(a) Candidate vs retained champion')
benchmark_panel(axes[1], '(b) Same frozen original opponent')
fig.tight_layout(pad=.8, w_pad=1.7)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/evaluation.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(6.65, 2.3))
for ax, key, title in zip(axes, ['loss', 'policy_loss', 'value_loss'],
                          ['Total data loss', 'Policy cross-entropy', 'Value MSE']):
    values = [r['epochs'][-1][key] for r in rows]
    assert np.isfinite(values).all()
    ax.plot(x, values, color=purple, linewidth=1)
    ax.set(xlabel='Candidate iteration', ylabel='Training loss', xlim=(0, int(x[-1])))
    ax.grid(alpha=.18)
fig.tight_layout(pad=.8, w_pad=1.7)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/optimization.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

# Reuse the trend fit's opening indices in the diagnostic comparisons.
candidate_labels = [str(i - start_iteration) for i in benchmark_iterations]
positions = np.arange(len(benchmarks))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(6.65, max(3.15, .25 * len(benchmarks) + 1)), sharey=True,
                         gridspec_kw={'width_ratios': [1, 1.15]})
left = np.zeros(len(benchmarks))
for key, label, color in [('wins', 'Wins', teal), ('draws', 'Draws', '#dfe5eb'),
                           ('losses', 'Losses', '#b15b42')]:
    counts = np.array([b[key] for b in benchmarks])
    axes[0].barh(positions, counts, left=left, height=.7, color=color, label=label)
    for row, count, offset in zip(positions, counts, left):
        if count:
            axes[0].text(offset + count/2, row, str(count), va='center', ha='center',
                         fontsize=7, color='#253044' if key == 'draws' else 'white')
    left += counts
assert np.all(left == 128)
axes[0].set(title='(a) Win / draw / loss counts', xlabel='Games', xlim=(0, 128),
            xticks=[0, 32, 64, 96, 128], yticks=positions, yticklabels=candidate_labels,
            ylabel='Promoted candidate iteration')
axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=3,
               frameon=False, handlelength=1, columnspacing=1)
for seat, offset, color, marker in [(0, .13, teal, 'o'), (1, -.13, purple, 's')]:
    means = seat_scores[:, seat].mean(axis=1) * 100
    intervals = np.array([np.quantile(values[bootstrap_indices].mean(axis=1), [.025, .975])
                          for values in seat_scores[:, seat]]) * 100
    assert np.all(intervals[:, 0] <= means) and np.all(means <= intervals[:, 1])
    axes[1].errorbar(means, positions + offset,
                     xerr=[means-intervals[:, 0], intervals[:, 1]-means],
                     fmt=marker, color=color, capsize=2, markersize=3.5, linewidth=1,
                     label=f'Player {seat + 1}')
axes[1].axvline(50, color=grey, linestyle='--', linewidth=.8)
axes[1].set(title='(b) Score by player assignment', xlabel='Score (%)',
            xlim=(20, 85), xticks=[20, 40, 60, 80])
axes[1].grid(axis='x', alpha=.18)
axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.16), ncol=2, frameon=False)
fig.tight_layout(pad=.8, w_pad=1.5)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/benchmark-outcomes.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.65, max(2.7, .18 * len(benchmarks) + 1.4)))
im = ax.imshow(pair_scores, cmap='RdBu', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
ax.set(yticks=np.arange(len(benchmarks)), yticklabels=candidate_labels,
       ylabel='Candidate iteration', xticks=[0, 8, 16, 24, 32, 40, 48, 56, 63],
       xlabel='Opening pair (fixed ID; same order for every candidate)')
ax.set_yticks(np.arange(len(benchmarks) + 1) - .5, minor=True)
ax.grid(which='minor', axis='y', color='white', linewidth=1)
ax.tick_params(which='minor', left=False)
colorbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.09, pad=.28,
                       ticks=[0, .25, .5, .75, 1], aspect=45)
colorbar.ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
colorbar.set_label('Mean score across both player assignments', fontsize=7)
fig.tight_layout(pad=.8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/opening-performance.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

differences = np.diff(pair_scores, axis=0)
means = differences.mean(axis=1) * 100
intervals = np.array([np.quantile(values[bootstrap_indices].mean(axis=1), [.025, .975])
                      for values in differences]) * 100
assert np.allclose(means, np.diff([b['score'] for b in benchmarks]) * 100)
assert np.all(intervals[:, 0] <= means) and np.all(means <= intervals[:, 1])
limit = max(10, np.ceil(np.abs(intervals).max()))
fig, ax = plt.subplots(figsize=(6.65, max(2.8, .3 * len(means) + .8)))
for index, (point, (low, high)) in enumerate(zip(means, intervals)):
    row = len(means) - 1 - index
    color = purple if benchmark_iterations[index + 1] == best_benchmark_iteration else '#253044'
    ax.errorbar(point, row, xerr=[[point-low], [high-point]], fmt='o', color=color,
                capsize=3, markersize=4, linewidth=1.2)
    ax.text(limit + 2.3, row, f'{point:+.1f}', ha='right', va='center', fontsize=7, color=color)
ax.axvline(0, color=grey, linestyle='--', linewidth=.8)
ax.set(xlabel='Change in score against frozen original (percentage points)',
       ylabel='Earlier → later candidate', xlim=(-limit, limit + 3),
       ylim=(-.7, len(means)-.4), yticks=np.arange(len(means))[::-1],
       yticklabels=[f'{a} → {b}' for a, b in zip(candidate_labels, candidate_labels[1:])])
ax.grid(axis='x', alpha=.18)
fig.tight_layout(pad=.8)
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/paired-improvements.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)

pilots = data['pilots']
assert [p['setting'] for p in pilots] == ['old_reset', 'persistent_3e5', 'persistent_8e5']
assert all(p['heldout_games'] == 384 for p in pilots)
assert all(p['baseline_heldout'] == pilots[0]['baseline_heldout'] for p in pilots)
fig, axes = plt.subplots(1, 2, figsize=(6.65, 2.65))
for pilot, color, marker, label in zip(pilots, [grey, teal, purple], ['s', '^', 'o'],
        ['Reset / blended, 8e−5', 'Persistent / raw, 3e−5', 'Persistent / raw, 8e−5']):
    assert [r['round'] for r in pilot['rounds']] == [1, 2, 3, 4]
    for ax, key, title in zip(axes, ['policy_ce', 'value_mse'],
                              ['(a) Held-out policy cross-entropy', '(b) Held-out value MSE']):
        values = [pilot['baseline_heldout'][key]] + [r['heldout'][key] for r in pilot['rounds']]
        assert np.isfinite(values).all()
        ax.plot(range(5), values, marker=marker, markersize=3.5, color=color,
                 linewidth=1.3, label=label)
        ax.set(title=title, xlabel='Pilot round (0 = shared baseline)', ylabel='Prediction loss',
               xticks=range(5), xlim=(-.1, 4.1))
        ax.grid(alpha=.18)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=6.5)
fig.tight_layout(pad=.8, w_pad=1.8, rect=(0, .1, 1, 1))
for suffix in ('pdf', 'png'):
    fig.savefig(ROOT / f'figures/pilot-learning.{suffix}', dpi=220, bbox_inches='tight')
plt.close(fig)
print(f'Validated {len(rows)} persistent candidates (0–{x[-1]}), {len(promotions)} promotions, '
      f'{len(benchmarks)} fixed benchmarks and all {len(benchmarks)*128} per-game outcomes; '
      'generated PDF/PNG figures, including four additional diagnostic figures.')
