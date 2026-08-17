# Transcript cross-check — 2026-08-12 supervisor meeting

Source: Fathom recording of the Jahid ↔ Ahmmad O.M. Saleh meeting
(fathom.video/calls/782792212), cross-checked 2026-08-17 by a read-only agent
against `Plan after the meeting.md` and ADRs 0006/0007/0008. Timestamps link
into the recording. Caveats: Fathom's speaker labels are scrambled in places
(attribution below is by content); ASR garbling ("music" = MuSiQue,
"Quen" = Qwen, "G-Rack" = GRAG, "hope" = hop).

This note records what the transcript supports. It changes no decision;
discrepancies are triaged in the companion GitHub issue.

## Verdicts on the recorded decisions

| Repo claim | Verdict |
|---|---|
| Supervisor meeting (Jahid + Saleh) | Confirmed by content; the transcript itself carries no date |
| Jury dropped, engineering-not-research reasoning (ADR 0006 §1) | Confirmed; "moves to Future Work" and the CLAUDE.md closed-model tie-in are repo glosses; the drop was Jahid's concession to Saleh's stated opinion ([1:12:34] "If you are sure about the judge idea… I will continue with you"), not an edict |
| Dataset roles: MuSiQue = decomp + e2e, MetaQA = e2e via GRAG (ADR 0006 §2) | Confirmed ([1:14:00]–[1:14:29]); ADR omits a sanctioned fallback: [1:15:24] "If you want to shrink… to one dataset, continue with the music" |
| Few-shot method fixed: bi-encoder top-20 → cross-encoder top-5, typed masking (ADR 0006 §3) | Method + typed masking confirmed ([1:24:27] "Keep the experiment on bi-encoder, cross-encoder type… leave the rest as an ablation"); **the numbers 20 and 5 were not fixed in the meeting** — top-20 is v1 practice ([42:55]), top-5 appears only in hypotheticals ([38:17], [1:18:00]) |
| Pool size 2000 (ADR 0006 §4) | Confirmed verbatim ([1:24:27] "2000 was the best one. So keep it 2000", reconfirmed [1:32:24]); note the transcript is internally inconsistent on the evidence — [26:16]/[47:17]/[1:29:25] describe 4000 as (co-)best |
| Contribution framing (ADR 0006 §5) | Confirmed ([1:11:38] "You are proposing a strategy, how you choose your few shots… with empirical study"); "hop-aware" wording is a gloss |
| Router OPEN pending guided-vs-unguided (ADR 0006 §6) | Confirmed as end state ([1:16:12]–[1:16:23]); preceded by a near-removal moment ([58:33] "will you include router or not? No… 90% I'm sure") that was then reopened |
| Length cap not cheating (Prompt 4) | Confirmed ([36:22] "you may set a stopping criteria… fix the max token… It's not cheating"); supervisor's example cap was **6** lines, the plan's default is 8 (config) |
| Eval set 600 = 200/hop (ADR 0007) | Confirmed ([26:16], [1:28:51] "I will pick the 600 questions that I have checked"; [1:29:13] fine-tuning evaluated "on the same 600 samples"); Jahid voiced unresolved doubt about the size ([28:50]) |
| No 600M router cap; up to 8B (ADR 0008) | Transcript **silent on any cap**; router work at Qwen-3B/7B and Mistral-7B discussed approvingly ([10:30], [12:29], [24:18]); the only "600 million" is an unrelated competition anecdote ([1:34:05]); quantization never mentioned. Consistent with ADR 0008's provenance finding; its supersession caveat stands |

## The two open items from issue #6 — confirmed still open

- **Thesis-primary metric: explicitly deferred.** [31:59] "For the other
  metric, we will leave it for the next meeting." [32:51] "It's not studied
  fully, so we don't know what is the best evaluation metric can be."
- **Embedding model: never chosen.** The bi-encoder/cross-encoder *method* was
  fixed; no specific model was named ([42:26] mentions "MiniLM and stuff like
  that" descriptively; [43:52] pre-trained, nothing trained).

## Said in the meeting, recorded nowhere (for Jahid to triage)

1. **Router as regression, not only classification** ([1:02:00] "you can deal
   with it as a regression problem… You are putting an assumption that you
   already know the maximum number of hops").
2. **Apply the same few-shot/pool method to the router itself** if kept, and
   evaluate MuSiQue with-router vs without ([1:02:00], [1:15:24]).
3. **Reinforcement learning out of scope for the master's** ([1:34:39] "this
   idea is worth your PhD… I don't put it as a baseline").
4. **Check the handmade composite score against standard re-ranking methods /
   bias check** ([46:18] "This composite score is handmade… Maybe it's
   biased. Let's put a note").
5. **Literature check before adopting any method** ([1:07:16]).
6. **Mixing MetaQA + MuSiQue training data** floated, "not yet decided"
   ([1:33:23], [1:34:03]).
7. **Fine-tuning sequencing ambiguous**: Jahid asked to fine-tune first
   ([1:28:00]); Saleh recommended pipeline-first ([1:30:28]) but closed with
   "but do fine-tuning. Let's see" ([1:29:56]) and "In case you do the fine
   tuning and everything didn't crash… continue following the figure I shared"
   ([1:35:43]). The plan resolves this silently by putting fine-tuning last.
8. **The supervisor asked for "a t-test, a statistical test"** ([31:33]); the
   metrics work implements paired bootstrap CIs + McNemar instead — a
   defensible substitution, recorded here as one.
9. **Error asymmetry value judgment**: over-decomposition more tolerable than
   under-decomposition ([33:23] "a three-hop question, it's fine to make it a
   four-hop, but it's not fine to make it a two-hop… check how many questions
   has been… under-decomposed, not over-decomposed").
10. **Failure cases kept in the ablation study** ([1:16:58]).
11. **The pipeline figure Saleh screenshotted and said he would send**
    ([1:33:00]) is referenced nowhere in the repo.
12. Presentation action items (dataset-statistics slide, ROUGE explanation,
    define typed/uniform/raw, un-bold non-best numbers) — [09:16], [27:42],
    [54:00].

## GRAG, fine-tuning, timeline

- **GRAG**: the supervisor's method ([33:55] "you can connect it to my
  method, G-Rack"); the operative line [1:14:13] "We can run MetaQA data on
  GRAG **that you have**" is ambiguous about who has it; **no handover date,
  repo, or access mechanism was discussed**. Prompt 8's
  external-dependency-to-chase stance is the safe reading.
- **Fine-tuning was ordered, not merely allowed** ([1:29:56] "but do
  fine-tuning"), LoRA named, two arms (best pool / full train split)
  ([1:27:05]), expected "a couple days" ([1:32:24]), both outcomes sellable
  and never hidden ([1:30:17]–[1:31:36]).
- **Closed-model-judging rule corroborated**: [49:29] "you said that Gemini is
  not kind of truth… It may hallucinate, so we cannot use it as a judge";
  [1:10:53] "GPT, Gemini, these ones, closed, big models… I don't like this
  approach. On research."
- The meeting decisions were to be **presented to the professor ("hoca")**
  ([1:26:25] "We present it to… hoca, what is our plan for ablation study,
  what is our baseline, and we will continue based on that").
- No mention of the October/November/December calendar in the transcript.
