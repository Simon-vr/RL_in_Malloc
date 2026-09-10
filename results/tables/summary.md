# Evaluation summary

Mean over rounds. HHI follows the paper convention (`sum((l_i/S)^2)`, higher = more concentrated).

| distribution | policy | occupancy | duration | fragmentation | hhi | hhi_complement |
|---|---|---|---|---|---|---|
| lognormal_train | DQN Agent | 0.9489 | 117.28 | 0.6882 | 0.1798 | 0.8202 |
| lognormal_train | First-Fit | 0.9535 | 117.86 | 0.6701 | 0.1921 | 0.8079 |
| lognormal_train | Best-Fit | 0.9588 | 118.59 | 0.6239 | 0.2311 | 0.7689 |
| lognormal_train | Worst-Fit | 0.7838 | 99.22 | 0.8723 | 0.0851 | 0.9149 |
| lognormal_large | DQN Agent | 0.9098 | 33.97 | 0.5538 | 0.3230 | 0.6770 |
| lognormal_large | First-Fit | 0.9111 | 33.99 | 0.5504 | 0.3262 | 0.6738 |
| lognormal_large | Best-Fit | 0.9152 | 34.14 | 0.5277 | 0.3449 | 0.6551 |
| lognormal_large | Worst-Fit | 0.8115 | 30.84 | 0.7015 | 0.2269 | 0.7731 |
| uniform | DQN Agent | 0.8857 | 20.52 | 0.4581 | 0.4321 | 0.5679 |
| uniform | First-Fit | 0.8879 | 20.59 | 0.4483 | 0.4419 | 0.5581 |
| uniform | Best-Fit | 0.8929 | 20.71 | 0.4366 | 0.4508 | 0.5492 |
| uniform | Worst-Fit | 0.8239 | 19.22 | 0.5529 | 0.3654 | 0.6346 |
| bimodal | DQN Agent | 0.9026 | 54.19 | 0.4558 | 0.4080 | 0.5920 |
| bimodal | First-Fit | 0.9064 | 54.45 | 0.4237 | 0.4425 | 0.5575 |
| bimodal | Best-Fit | 0.9176 | 55.03 | 0.3768 | 0.4921 | 0.5079 |
| bimodal | Worst-Fit | 0.7701 | 47.55 | 0.6734 | 0.2389 | 0.7611 |
