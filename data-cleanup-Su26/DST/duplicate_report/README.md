# Duplicate Detection: Output Guide

This document explains the output from running all of the finding_dups notebook.

---

## What the code does

The routine removes duplicates in two passes. **The Timestamps are a column here**.

**Case one — exact duplicates.** `drop_case_one_dups(df)` treats two rows as
duplicates only if they are identical across *every* column. It keeps the first
occurrence of each repeated row and discards the rest. It returns two frames: the
deduplicated frame, and a frame containing *every* row that belongs to a
duplicated group — all occurrences — via
`duplicated(keep=False)`.

**Case two — duplicates ignoring `Flag`.** `drop_case_two_dups(df, ignore_col='Flag')`
treats two rows as duplicates if they are identical across every column *except*
`Flag`. It is run on the output of case one, i.e. on a frame that already has no
exact duplicates. It returns the same pair of frames (deduplicated, plus all rows
in the duplicated groups). If `ignore_col` is not present in the frame it prints a
warning, makes no changes, and returns an empty "duplicates" frame.

Because case two runs *after* case one has removed every exact duplicate, any
group case two finds must consist of rows that are identical on all non-`Flag`
columns yet are not identical overall — and the only place left to differ is
`Flag`. So **case two specifically surfaces records that are identical except for
a conflicting flag.**

**Case three - only duplicate timestamps**. This is the most troubling case with no clear fix,
since who knows which measurement is correct; therefore, we only report the occurrences. 
Notably, this case **ONLY occurs in proximity to the end of Daylight Savings Time (DST) at transition time** and
a duplicate 1 am signature on the day DST ends would indicate the data observes DST; however,
there is **never an indication of both DST starting and ending in one single file** (investigate in DST_check notebook). Additionally, 
the vast majority of case three duplicates occur exactly a day after DST ends.

---

## Preconditions

A few things must hold for the numbers to mean what you expect:

- **The index is reset first.** `drop_duplicates` and `duplicated` look only at
  columns, never the index. The pipeline calls `reset_index()` so the timestamp
  becomes a column and participates in duplicate detection. This is correct when
  the index is a meaningful key such as a timestamp with possible repeats. It
  would be *wrong* if the index were a plain `RangeIndex`, because that injects a
  unique `0..n-1` column and then no row can ever be a duplicate.
- **Missing values match each other.** Both functions inherit pandas' rule that
  `NaN`, `NaT`, `None`, and `pd.NA` are treated as equal to themselves when
  comparing rows. Two rows that are both missing in the same column (and equal
  elsewhere) count as duplicates rather than being skipped.
- **Order matters.** Case two consumes case one's *de-duplicated* output. Running
  it on the raw frame instead would fold the exact duplicates in with the flag
  conflicts and inflate the case-two counts.

---

## How to read each printed line

### Case One Report
| Printed line | What it counts |
|---|---|
| Original DataFrame rows | Rows in the raw frame (after `reset_index`). |
| After dropping case one duplicates | Rows left once exact (all-column) duplicates are collapsed to the first occurrence. |
| Difference after case one duplicate removal | Exact-duplicate rows removed — the redundant copies. |
| Rows containing all case one duplicates | Every row that is part of some repeated group (all occurrences). |
| The difference between all and unique duplicates | Should be equitant to setting keep = 'first' or the previous difference |

### Case Two Report
| Printed line | What it counts |
|---|---|
| Unique rows among case one duplicates | Distinct records that were repeated at least once. |
| After dropping case two duplicates | when ignoring `Flag`, Rows left once duplicates are also collapsed. |
| Difference … case two duplicate removal | when ignoring `Flag`, the redundant rows removed. |
| Rows containing all case two duplicates | Every row in a group that is identical except for `Flag`. |
| Unique rows among case two duplicates | Distinct records that had a flag conflict. |
| The difference between all and unique duplicates | Should be equitant to setting keep = 'first' or the previous difference |

---

## What you should expect if the code is correct

Let **R** = the number of rows reported as "containing all duplicates"
(`duplicated(keep=False)`), and **K** = the number of unique rows among them
(`drop_duplicates` of that frame). The following hold for each case by
construction, so they double as correctness checks:

1. **Row counts never increase:** `original ≥ after-case-one ≥ after-case-two`.
2. **Removed = original − deduplicated**, and it is `≥ 0`.
3. **Removed = R − K.** `drop_duplicates` deletes one fewer row than the size of
   each duplicated group, so `Σ(group_size − 1)` = (rows in duplicated groups) −
   (number of duplicated groups) = `R − K`. If "rows containing all duplicates"
   minus "unique rows among them" does not equal the removed count, something is
   wrong.
4. **R ≥ 2K**, with equality exactly when every duplicated group has size 2.
   `R = 2K` therefore tells you the duplicates come strictly in pairs.
5. **Reconciliation:** rows appearing exactly once number `original − R`; adding
   the `K` distinct repeated records gives the deduplicated total:
   `(original − R) + K = deduplicated`.

---

## Worked example: the CB01 run

**Case one**

| Quantity | Value |
|---|---|
| Original rows | 85,252 |
| After case-one dedup | 81,185 |
| Removed (difference) | 4,067 |
| Rows in duplicate groups (R₁) | 6,191 |
| Distinct repeated records (K₁) | 2,124 |

| Check | Computation | Result |
|---|---|---|
| removed = original − deduped | 85,252 − 81,185 | 4,067 ✓ |
| removed = R₁ − K₁ | 6,191 − 2,124 | 4,067 ✓ |
| R₁ ≥ 2·K₁ | 6,191 ≥ 4,248 | ✓ (not all pairs) |
| reconcile | (85,252 − 6,191) + 2,124 | 81,185 ✓ |

Reading: 2,124 distinct records were each repeated, together accounting for 6,191
row-instances; 4,067 of those are redundant copies that get removed. Equivalently,
79,061 rows appear exactly once, and adding back one copy of each of the 2,124
repeated records gives 81,185.

**Case two** (run on the 81,185-row case-one output, ignoring `Flag`)

| Quantity | Value |
|---|---|
| Input rows | 81,185 |
| After case-two dedup | 81,177 |
| Removed (difference) | 8 |
| Rows in duplicate groups (R₂) | 16 |
| Distinct non-`Flag` records (K₂) | 8 |

| Check | Computation | Result |
|---|---|---|
| removed = input − deduped | 81,185 − 81,177 | 8 ✓ |
| removed = R₂ − K₂ | 16 − 8 | 8 ✓ |
| R₂ = 2·K₂ (all pairs) | 16 = 16 | ✓ → 8 flag-conflict pairs |
| reconcile | (81,185 − 16) + 8 | 81,177 ✓ |

Reading: because `R₂ = 2·K₂`, the case-two duplicates are exactly 8 *pairs*. Each
pair is identical on every column except `Flag`, with two different `Flag` values
— i.e. the same underlying record carrying conflicting flags. Collapsing each pair
to its first occurrence removes 8 rows, leaving 81,177. These 8 pairs are a
data-quality signal worth inspecting before they are collapsed blindly.

---

## Signs something is wrong

- **R − K ≠ removed** for either case — the core red flag.
- Any stage's row count **exceeds** the previous stage's, or a **negative**
  difference appears.

