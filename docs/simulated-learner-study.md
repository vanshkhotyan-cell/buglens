# Phase 5 simulated learner study

Team InferenceX · December 1 midway review

## Design and reproduction

From the original checkout, run the commands below. The configuration file holds the population, tutor, scenarios, cohort size, sessions, per-session budget and seeds. The module’s defaults reproduce the shipped configuration. Use a different output directory with `--out` for an exploratory run. All defaults are engineering choices.

```sh
cd /Users/mohdfarman/BugLens/buglens
uv sync --locked
.venv/bin/python -m experiments.sim_learners --config reports/simulation/config.json
.venv/bin/python -m pytest -q
.venv/bin/ruff check app/streamlit_app.py experiments/sim_learners.py tests/test_sim_learners.py
.venv/bin/ruff format --check app/streamlit_app.py experiments/sim_learners.py tests/test_sim_learners.py
uv lock --check --offline
git diff --check
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Select **Research analytics** in the sidebar. Start with the three baseline rows, open **Model assumptions and paired differences**, then select **high_relapse**. The table follows the selector. The three figures display every scenario for comparison. A missing summary or chart produces an unavailable message.

## Counterfactual learners

Each of three seeds (17, 41, 83) generates 100 independent synthetic profiles. Copy each profile into all three conditions. Reuse corresponding random streams across conditions for initial states, session relapse, assessment noise and activity opportunities. The scenario comparisons also reuse seeds. A learner shares a true learning rate and relapse rate across five families. Initial active states are independent Bernoulli draws with probability 0.5. Draw each learner’s two rates independently from Beta(mk, (1−m)k), with concentration k = 15 and baseline means 0.32 and 0.04. An endpoint mean gives a constant rate. Each inactive family can reactivate independently at the start of every session after the first.

For any item, the probability of a correct response is 0.12 when the family is active and 0.90 when inactive. Practice and review then give an active family one chance to resolve at the learner’s own rate, regardless of its observed answer. Diagnostic probes and assessments produce no learning. All conditions use these same rules. One simulated day separates sessions. Assessment uses two distinct SymPy-checked five-item forms with one held-out item per family, a shared pre-test, and paired post-test noise. Item difficulty is constant within a family. Scores are fractions correct.

## Tutor and activity allocation

The targeted policy receives only the item and observed response category. Its independent inverse-BKT estimates start at 0.5, with assumed accidental-correct and slip probabilities of 0.10, learning of 0.30 and relapse of 0.05. An isolated in-memory SQLite connection runs the existing tracer and review queue with a simulated clock. Synthetic records never enter persistent learner memory.

The first of three slots buys a diagnostic probe. For a family with estimated active probability p and conditional correct-response probabilities a and b, binary expected information gain in bits is H(pa + (1−p)b) − pH(a) − (1−p)H(b), where H is binary entropy. Choose the maximum and break ties with a seeded permutation. Snapshot eligible reviews at session start, then use subsequent slots for queued reviews in interleaved order and otherwise target the highest current marginal. Targeted practice uses the existing cognitive-conflict lesson builder and verified practice example. Both active arms use the authored practice bank and the same response and learning model. Random practice samples a family and one of its two practice items uniformly. Control receives both assessments and experiences only the shared relapse process between them.

Probes cost time even when their evidence does not change the next decision. A targeted lesson, a review, and a random question each cost one slot. The study matches item opportunities; real elapsed time, explanation quality, self-explanation, and engagement are unmeasured. The tracer follows its existing correct-completed-practice update even though hidden learning can occur after either response. Independent marginals remain independent. The six-class adapter requires a distinct diagnostic prior and evidence contract.

## Outcomes and uncertainty

Normalized gain is (post − pre)/(1 − pre) per learner. Exclude perfect pre-tests only from gain and report the eligible count. Negative gains are retained. Error recurrence divides the number of families answered incorrectly on both forms by all pre-test family errors. This observable can include slips. A zero denominator gives a missing estimate. Summaries include score and gain means, medians and sample variances.

For each initially active family, record the first session ending inactive. Families still awaiting a first resolution at the study horizon are administratively censored. The curve gives the fraction still awaiting first resolution. Its median is the first session at or below 50%, if reached. Restricted mean sessions average the event time capped at the horizon over all initially active families. The separate active-at-end count includes families that resolved and relapsed. Only the evaluator uses latent states for these endpoints.

Percentile intervals use 1,000 bootstrap resamples of whole learner profiles (fixed seed 504). The same profile indices are used for each arm, preserving paired differences, within-learner family dependence and common assessment noise. Intervals describe simulation sampling variation under each fixed scenario. They omit uncertainty about the assumed model, calibration and transfer to classrooms. The scenarios are descriptive sensitivity checks without multiplicity adjustments.

## December review results

Across 300 matched baseline profiles per arm, the pre-test mean was 0.494. The targeted, random and control post-test means were 0.792, 0.777 and 0.424. Their normalized gains were 0.528, 0.489 and −0.260 across the 290 eligible profiles in each arm. The targeted-minus-random gain was 0.039 (paired 95% interval −0.001 to 0.078). Their recurring errors numbered 181, 207 and 625 out of 759 initial errors. Of 762 initially active families, 97, 132 and 762 were censored. Both active arms reached median first resolution at session 4. The control median was unreached.

Increasing mean relapse to 0.12 changed the targeted-minus-random gain to −0.074 (−0.127 to −0.020). Raising slip to 0.22 and accidental correctness to 0.25 gave −0.013 (−0.054 to 0.029). Under this world model the control has no route to new learning and can accumulate misconceptions. Its declining scores reflect that assumption. These synthetic comparisons cannot establish real educational efficacy.

Under `reports/simulation/`, `learners.csv` holds one synthetic learner/seed/arm/scenario row, `aggregate.csv` summarizes groups, `sensitivity.csv` gives paired contrasts, and `summary.json` includes the configuration, intervals and resolution curves. The three PNGs show recurrence, mean normalized gain and time to first resolution. No human data were collected.

## Live integration boundaries

The study exercises authored remediation content, bounded answer classification, BKT and review scheduling. It models responses to explanations probabilistically. Live replay from arbitrary student work, neural inference with calibrated likelihoods, and a validated mapping between longitudinal marginals and the six-class diagnostic model remain open. Family independence, equal item difficulty, simplified feedback costs, no spontaneous recovery, fixed response noise and single-session resolution limit interpretation. Lasting resolution, cross-family transfer and human learning require separate validation.
