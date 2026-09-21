# Protocol (placeholder)

The source document `loss-function-followup-protocol.md` was not present in the
repo root when the scaffold was built, so this file is a stand-in assembled from
the scaffold brief. Replace it with the real protocol via
`git mv loss-function-followup-protocol.md docs/protocol.md` when it is available.

## Experiments

- **Exp 1, temperature sweep.** `minilm` and `biomedbert-fulltext`, MNRL loss,
  tau in {0.01, 0.05, 0.2, 1.0}, random within-subset pairs.
- **Exp 2, loss family.** `biomedbert-fulltext`, losses {MNRL, triplet, CoSENT,
  MLM}, tau 0.05.
- **Exp 3, data vs loss.** `biomedbert-fulltext`, MNRL, tau 0.05, pair sets
  {matched, mismatched, random} by standardized textstat distance. Compared to
  the MedCPT reference row from the first post.
- **Exp 4, topological auxiliaries.** `biomedbert-fulltext`, MNRL + lambda * aux,
  aux in {textstat head, distance preservation, persist0 H0}, lambda in {0.1, 1.0}.

Every checkpoint at fractions {0, 0.1, 0.25, 0.5, 1.0} of training is evaluated
on both corpora with the bakeoff's three layers plus the router layer, producing
one CSV row per (model, loss, tau, pair_set, aux, lambda, checkpoint, corpus).

## Reproduction check

Before any fine-tune is trusted, evaluate `ckpt_0.0` of `biomedbert-fulltext` and
`minilm` on both corpora and confirm the Layer 1-3 numbers match the first post's
rows for those encoders within bootstrap noise. See `docs/harness-notes.md` §8 for
the MedCPT reference values and §1 for why the eval sample must be byte-identical.
