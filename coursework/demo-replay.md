# Agent demonstration replay

[Rendered agent demonstration](agent-performance-demo.mp4), 64.6 seconds.

Two examples from the fixed final opening set. Both use Double DQN seed 51 as
player 1. Columns below are zero-indexed. The video displays columns 1–7. Use the full trace files in
`study-evidence/dqn-study/evaluation/ddqn-51/` to animate the actual games.
Present these as selected examples and display the aggregate results alongside them.

## Learning from fresh weights

Use opening ID 2, player 1, against uniform random play. Compare
`final-network-0-random.json` with `final-network-5000-random.json`.

- Begin both replays after the shared opening `2, 6, 2, 3, 5, 2, 4, 2`.
- The untrained policy first plays column 6 and loses. The trained policy first
  plays column 3, later blocks the third opposing counter in column 2, and wins.
- Label each panel with the training-game count. Include the aggregate initial
  and final learning results so the example has context.

## A tactical failure and its correction

Use opening ID 7, player 1, against depth-two minimax. Compare
`final-network-5000-minimax.json` with `final-guarded-5000-minimax.json`.

- The opening is `6, 4`. Advance both panels through their shared continuation
  `3, 3, 6, 6, 4, 6, 0, 2, 3, 4, 3, 6`.
- Freeze just before the next player-1 action. The opponent has three
  consecutive vertical counters in column 6 and can win immediately there.
- Greedy Q selects column 5 and minimax wins on its next move. The tactical
  filter permits only column 6. The guarded policy eventually wins this replay.
- Label this comparison as the same weights with two inference rules. The
  tactical filter is a rule-based safeguard.

Allow about 10 seconds of context, 40 seconds for the before/after learning
replay, 40 seconds for the tactical example, and 15 seconds for the final
comparison and its limitations. Keep the finished video under two minutes.
