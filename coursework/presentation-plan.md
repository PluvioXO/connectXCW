# CM32032 group presentation recording plan

Target runtime **3:35–3:50** at natural speaking pace. Every group member
speaks. Use the six beats below as a shared script and assign them before
recording. With four speakers, combine beats 1–2 and 5–6. With five, combine
beats 5–6. With six, give each person one beat. Add each speaker's name when the
group approves the allocation. Use the separate [agent demo](agent-performance-demo.mp4)
for the performance video submission.

| Time | What to show | Suggested spoken content |
|---|---|---|
| 0:00–0:32 | The board and a drop. Show the 4.53 trillion position estimate. | “We study Connect Four, an alternating zero-sum game. A move drops a counter into one of seven columns. Four in a row wins. The game is solved, but a dense action-value table would take about 127 terabytes. We asked what a small network could learn from self-play and what still required search.” |
| 0:32–1:08 | The Double DQN equation from the Method section. Animate the player swap between two states. | “Our implementation uses replay and two networks. The online network picks a legal successor move and the target network evaluates it. We subtract that estimate because the next player is the opponent. Terminal moves use their actual outcome. Starting from fresh weights, both DQN variants improved from about 70 to 95 percent against random play, but reached only about 11 percent against shallow minimax.” |
| 1:08–1:44 | The move-matched learning curves. Highlight the five seed pairs and fixed budget. | “The network often left an immediate winning reply. We added a loss on actions whose outcomes we could certify within two plies. We ran five matched seed pairs at eighty thousand moves each, with the same optimiser update count. On separate openings, the standard network scored 11.9 percent against depth-two minimax and the tactical variant scored 9.8 percent.” |
| 1:44–2:16 | Solved endgames beside the shared defensive-position results. | “On solved late-game defences, tactical supervision raised optimal move selection from 46.9 to 51.7 percent. We then gave networks from ten seeds the same 1,797 earlier defensive positions from match traces. Tactical supervision raised the rate of leaving an immediate loss from 38.3 to 42.9 percent there. This shows why the choice of test positions matters.” |
| 2:16–2:52 | The lookahead table and screenshots from the separate demonstration. | “We fixed the two-ply tree and tested it on 128 fresh openings in both seats. The Q-network rose from 16.4 percent at initialisation to 30.4 percent after training. A hand-coded window evaluator scored 39.5 percent with the same tree. The first comparison measures learning. The second shows the gap to a stronger baseline.” |
| 2:52–3:34 | The horizon comparison and the scope of the inherited baseline. | “Our agent learns from terminal rewards. Over ten paired seeds, one step scored 10.9 percent and up to three scored 9.7 percent at equal budgets. The interval spans zero when we vary both training seeds and openings. Our pretrained search agent is a separate deployment reference. These controls show where the fresh-weight learner improves and where it still falls short.” |

At rehearsal, ask each speaker to explain one design choice in their own
words, including the negative TD sign, why seeds are paired, the definition of
an avoidable immediate loss or why the search comparison uses the same tree.
Cut a sentence if the recording approaches four minutes. Keep the
exact score and identity out of the spoken Kaggle claim until the submission
record is attached. Confirm the final file opens, plays at normal speed and
contains all group voices.
