# How to actually score higher on the OpenEnv Arena — an evidence-based reflection

Written after Finance Desk v1 scored **2/40** (office 2, everything else 0; delta −2 vs the untrained
4/40). This is not a one-off: it is the single most important thing to understand before spending another
slot.

## What the board data actually says (45+ evaluated runs)

1. **No run beats chance.** naklitechie's meta-analysis (#149) and rca-umb's permutation test (#188):
   across all evaluated runs, scores are indistinguishable from independent draws around the base model
   (permutation p ≈ 0.6). Nobody's environment design has produced a result outside the noise.
2. **The noise band is 1–5/40, and the mean trained delta is ≈ −1.** Training *on average makes the model
   slightly worse* on the private tasks than leaving it untrained (4/40). Zero-gradient control runs
   (null-control, kvbatch) still scored 1–2/40 — so the save/eval pipeline itself seems to cost ~2 points
   regardless of what you train.
3. **Points live in two domains.** Industrial behaves like ~2 coin-flip tasks and office like ~1 sure + 1
   coin-flip; software 1 shows up in ~29% of runs; **finance, natural science, math, cybersecurity and
   media are ~0 for everyone**, with the occasional fluke 1. The untrained model already gets industrial ~2
   + office ~2 = 4.
4. **Steady gradient tends to *erode* the base solves.** rycerzes (#160): the only runs that scored 4–5
   kept industrial 2 + office 2; runs with strong, sustained training signal tended to *lose* those base
   solves. Our v1 did exactly this — it trained hard (reward 0.5→0.79) and came out with office 2 but
   industrial **0**. We trained away a point we started with.
5. **The top score (8/40) is a best-per-domain composite of five noisy runs** (4,4,2,4,4), not one good
   run. The leaderboard keeps each account's best per domain across all its runs.

## The uncomfortable conclusion

You cannot reliably beat ~4–5/40 **in a single run by designing a cleverer environment.** The evidence is
overwhelming and consistent across a dozen serious entrants. v1 losing to 2/40 was the *expected* outcome,
not a failure of our environment specifically.

## What therefore actually raises the leaderboard number

**A. Repeated daily submissions (highest-confidence lever).** Best-per-domain accumulates across runs, so N
independent noisy draws strictly dominate one. This is literally how the #1 account reached 8/40. One run
per 24h, each a fresh draw, banking the best cell per domain. This is the strategy with the only track
record of producing top scores.

**B. Do not damage the base model's existing competencies (industrial + office).** Since those two domains
carry essentially all the reliable points, a run should *reinforce* the kind of plain, correct reasoning the
base model already does there — not push the policy into a narrow, over-optimized regime that forgets it.
Concretely: keep industrial/office tasks at difficulty the base model can already do (so training nudges
"keep doing this" rather than "learn a weird trick"), and don't over-train.

**C. Train the one plausibly-transferable skill: output-format discipline.** The arena scores a reply only
if it yields exactly one valid JSON object with the right keys. A meaningful fraction of the base model's
private-task attempts fail *purely on format* (prose/newlines, unescaped quotes in code strings — 14–18% of
episodes in others' calibration). Unlike domain knowledge, "always emit one clean exact-keyed JSON object"
is domain-neutral and should transfer to the private tasks. Single-step + inline files + exact grading
(v2) maximizes clean-JSON reps and removes the code-string quoting failure mode entirely.

**D. The biggest missing capability: calibration on the real Qwen3.8-27B.** Every strong entrant calibrates
task difficulty on the actual eval model (HF router, thinking off) and keeps only tasks that stay mixed
(~0.3–0.7 solve). We cannot — this HF token lacks Inference Providers access (403). Without it we are
guessing difficulty, which is why we can't deliberately hit the ~0.5 mixed-gradient sweet spot or verify we
aren't saturating/over-hardening. **Enabling Inference Providers on the HF account (PRO or pay-as-you-go
credits) is the single highest-leverage change available.**

## What v2 does with this (what we can control without 27B access)

- **Single-step, inline files, exact grading** — removes the main zero-reward failure mode and makes every
  episode a clean-JSON rep (lever C).
- **Reweight the task mix toward the domains that carry points** (office + industrial) at **easier levels**
  the base model can already do, so training reinforces rather than erodes them (lever B). Keep all eight
  domains for the best-per-domain draw, finance still well represented as the theme.
- **Moderate, not maximal, domain-overfitting.** Keep breadth and easier levels so no single hard family
  dominates the gradient.

## What v2 cannot do, honestly

v2 is a *better-shaped single draw*, not a method that beats the noise floor — nothing we can build is, per
the evidence. The real gains come from **(A) running it repeatedly day over day** and **(D) getting 27B
calibration access**. Everything else is rearranging variance.
