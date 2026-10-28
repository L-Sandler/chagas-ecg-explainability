# Defend log

Answers in my own words, written without looking at the code. Each links to the code or
decision it explains; when that changes, the answer is due for review.
Start each session by re-answering 2-3 earlier questions marked `weak` from memory.

| # | Area | Status | Last tried |
|---|------|--------|------------|
| 1 | preprocess | weak | 2026-10-28 |

---

## Q1. A 7 s CODE-15% record arrives as `4096 x 12`. Which steps, in order, and what shape comes out?

Source: [preprocess_signal](../../src/preprocess.py#L63-L89), decisions 0001, 0006, 0008.

**Attempt 1 (2026-10-28):** some kind of band filter on high and low frequencies; padding to a
consistent length. Gaps: most steps missing, and no output shape.

**My answer:**
