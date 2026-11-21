# Connect Four coursework evaluation plan

The existing results describe fine-tuning an inherited search agent. The
Double DQN pilot completed 5,000 self-play games with seed 51. Its results on
development openings are in [the report](report.pdf) and
[the pilot summary](evidence/ddqn-pilot-summary.json). The matched three-seed comparison
then applied the protocol below with a tactical inference ablation. See
[the study summary](evidence/dqn-study-summary.json).

- Test whether separating action selection and evaluation improves learning speed
  or final greedy playing strength. DQN is the control. The network, replay, exploration and game budget match
  across the methods. The three paired initialisation seeds are 51, 52 and 53.
  Moves and updates are recorded because games vary in length.
- Use the initial checkpoint and common game-count checkpoints. Evaluate against
  seeded random play and depth-two minimax with both seats for every opening.
  Report seed-level curves and score differences, and pair opening-level
  comparisons. Report uncertainty across training seeds for general learning claims.
- Choose the final checkpoint rule on development data. Reserve a final opening
  set with a different seed. Pass `--exclude-openings-from DEVELOPMENT.json`
  to exclude development boards by identity, including the empty board.
  Use the same exclusions for every final checkpoint and opponent.
- Use representative matched before/after positions in the agent video and
  explain a failure case. Use aggregate scores to substantiate examples.
  Record the Kaggle rank, score, date, account and matching submission as
  external evidence for the inherited deployable agent.

| Deliverable | Completion check |
|---|---|
| Report | Seven or fewer main pages in the supplied template. Include the learning curve, paired comparison, limitations and verified Kaggle details. |
| Presentation | At most four minutes at natural pace. Everyone speaks. Cover the question, algorithm, controls, result and limitation. |
| Agent video | At most two minutes. Identify before/after checkpoints and fixed opponent on screen. |
| Source ZIP | Include source, dependencies, run settings, hashes and evidence. Verify reproduction from a fresh extraction. |
| Contribution form | Agree the contribution split. Each member supplies their own reflection and verifies their GenAI disclosure. |

The move-matched follow-up compares plain and tactically supervised Double DQN
across seeds 61–65. The test pack fixes the primary score on 64 new openings,
solved positions and secondary checks. The separate lookahead comparison
uses another 24 openings and an initial-weight search control. See
[the follow-up evidence](evidence/tactical-followup-summary.json).

The ten-seed follow-up uses the same 128 opening pairs for matched leaf
evaluation against depth-three minimax and greedy horizon comparisons
against depth-two minimax. The defence diagnostic fixes the 1,797 earlier
boards and both orientations for all seeds. The five added seeds share
the plain one-step control across analyses. Report mean ± sample SD
across seeds and keep paired confidence intervals separate. See the
[matched controls summary](evidence/q-controls-summary.json).

The rendered [agent-performance demo](agent-performance-demo.mp4) uses the
earlier matched study. The [presentation plan](presentation-plan.md) is for
the separate video in which every group member speaks.
