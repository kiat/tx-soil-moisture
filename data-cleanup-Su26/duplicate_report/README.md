# Duplicate Detection: Output Guide

This document explains the output from running all of the finding_dups notebook.


## What the code does

The routine removes duplicates in three passes. **The timestamps are a column here**.

**Case one — full_row_duplicates.** `drop_case_one_dups(df)` treats two rows as
duplicates only if they are identical across *every* column. It keeps the first
occurrence of each repeated row and discards the rest.

**Case two — full_dup_but_flag** `drop_case_two_dups(df, ignore_col='Flag')`
treats two rows as duplicates if they are identical across every column *except*
`Flag`. It is run on the output of case one, i.e. on a frame that already has no
case one duplicates. If `ignore_col` is not present in the frame it prints a
warning, makes no changes, and returns an empty "duplicates" frame.

Because case two runs *after* case one has removed every exact duplicate, any
group case two finds must consist of rows that are identical on all non-`Flag`
columns yet are not identical overall — and the only place left to differ is
`Flag`. So **case two specifically surfaces records that are identical except for
a conflicting flag.**

**Case three - only_duplicate_timestamps**. This is the most troubling case with no clear fix,
since it is not clear which measurement is correct. Notably, this case **mostly occurs the day after the end of Daylight Savings Time (DST)** with the indication of DST ending (dup at 1 am); however, there is **NEVER an indication of both DST starting and ending in one single file** (shown in DST_check.ipynb).

**All cases are treated by only keeping the first occurance by order of appearance.**

---

## Duplicate statistics

Generated from the all-occurrence duplicate CSVs in this report. "Duplicate groups" counts distinct duplicated records; "rows logged" counts every occurrence saved here (kept + removed); "rows removed by dedup" is what a keep-first treatment would drop.

### Overall

- **Files scanned:** 39 (33 soil, 6 met)
- **Raw rows across all files:** 3,051,573
- **Rows removed by deduplication:** 178,546 (**5.85%** of all rows)
- **Rows remaining after dedup:** 2,873,027
- **Files with at least one duplicate:** 37 of 39 — only **FD21, FD23** are completely clean

| Case | Stations affected | Duplicate groups | Rows logged (all occ.) | Rows removed by dedup |
|---|---|---|---|---|
| 1 — full_row_duplicate | 37 / 39 | 94,891 | 273,334 | 178,443 |
| 2 — full_dup_but_flag | 8 / 39 | 40 | 80 | 40 (flag-merge) |
| 3 — only_duplicate_timestamps | 33 / 39 | 63 | 126 | 63 (keep-first) |
| **Total** | — | **94,994** | **273,540** | **178,546** |

Case 1 dwarfs the others: it accounts for 99.9% of all removed rows, while Cases 2 and 3 together remove just 103 rows. This is consistent with Case 1 being a compile/telemetry artifact rather than a sensor issue.

### Case 1 — copies per duplicated row

Most duplicated records appear **three times**, not twice:

| Copies per row | Duplicated rows (groups) |
|---|---|
| 2× (one extra) | 11,356 |
| 3× (two extra) | 83,518 |
| 4× (three extra) | 17 |


### Case 3 — DST clustering

| Case 3 sub-bucket | Pairs | Rows |
|---|---|---|
| day_after_dst_ends | 47 | 94 |
| when_dst_ends | 2 | 4 |
| any_other_time | 14 | 28 |

94 of 126 Case-3 rows (75%) fall on the day after DST ends.

### Per-station

% affected = duplicate groups (all three cases) ÷ raw rows. Stations are sorted alphabetically; "—" means none found.

| Station | Type | Raw rows | Case 1 dropped | Case 2 pairs | Case 3 pairs | Rows removed (dedup) | % raw affected |
|---|---|---|---|---|---|---|---|
| CB01 | soil | 85,252 | 4,067 | 8 | 2 | 4,077 | 2.503% |
| CB01_met | met | 15,598 | 1 | — | — | 1 | 0.006% |
| CB04 | soil | 93,365 | 7 | — | 2 | 9 | 0.010% |
| CB04_met | met | 99,010 | 199 | — | 2 | 201 | 0.203% |
| CB06 | soil | 116,511 | 22,925 | 5 | 2 | 22,932 | 9.917% |
| CB06_met | met | 74,792 | 200 | — | — | 200 | 0.267% |
| CB07 | soil | 82,737 | 3 | — | 2 | 5 | 0.006% |
| CB09 | soil | 80,512 | 3 | — | 1 | 4 | 0.005% |
| CB10 | soil | 90,319 | 6 | — | 2 | 8 | 0.009% |
| CB15 | soil | 20,302 | 1 | — | 1 | 2 | 0.010% |
| CB19 | soil | 68,121 | 8,163 | — | 1 | 8,164 | 6.139% |
| CB20 | soil | 28,427 | 1 | — | 2 | 3 | 0.011% |
| CB25 | soil | 103,156 | 8,163 | — | 3 | 8,166 | 4.056% |
| CB26 | soil | 78,621 | 1 | — | 1 | 2 | 0.003% |
| CB27 | soil | 90,936 | 8,162 | — | 2 | 8,164 | 4.599% |
| CB31 | soil | 100,969 | 8,163 | — | 2 | 8,165 | 4.143% |
| CB32 | soil | 77,683 | 6 | — | 2 | 8 | 0.010% |
| CB33 | soil | 95,162 | 8,164 | — | 2 | 8,166 | 4.397% |
| FD02 | soil | 101,767 | 8,163 | — | 2 | 8,165 | 4.110% |
| FD02_met | met | 100,117 | 202 | — | 2 | 204 | 0.204% |
| FD03 | soil | 79,670 | 9,399 | 8 | 1 | 9,408 | 6.469% |
| FD03_met | met | 109,337 | 23,125 | — | 1 | 23,126 | 10.740% |
| FD08 | soil | 106,470 | 108 | — | 2 | 110 | 0.103% |
| FD11 | soil | 101,737 | 8,163 | — | 2 | 8,165 | 4.112% |
| FD12 | soil | 48,219 | 1,614 | 1 | 2 | 1,617 | 3.353% |
| FD13 | soil | 94,477 | 8,159 | 6 | 2 | 8,167 | 4.430% |
| FD14 | soil | 80,886 | 8,161 | — | 2 | 8,163 | 5.169% |
| FD16 | soil | 88,182 | 4,573 | — | 3 | 4,576 | 4.551% |
| FD17 | soil | 33,287 | 1 | — | — | 1 | 0.003% |
| FD18 | soil | 93,060 | 10,427 | 2 | 1 | 10,430 | 5.709% |
| FD21 | soil | 791 | — | — | — | 0 | 0.000% |
| FD22 | soil | 82,459 | 2 | — | 3 | 5 | 0.006% |
| FD23 | soil | 30,955 | — | — | — | 0 | 0.000% |
| FD24 | soil | 79,371 | 10,410 | 9 | 3 | 10,422 | 6.690% |
| FD28 | soil | 49,255 | 2 | — | — | 2 | 0.004% |
| FD29 | soil | 91,291 | 8,163 | — | 2 | 8,165 | 4.582% |
| FD30 | soil | 98,005 | 8,163 | — | 2 | 8,165 | 4.268% |
| WC05 | soil | 83,633 | 1,175 | 1 | 2 | 1,178 | 1.409% |
| WC05_met | met | 97,131 | 198 | — | 2 | 200 | 0.206% |

**Heaviest Case-1 burden:** FD03_met (23,125), CB06 (22,925), FD18 (10,427), FD24 (10,410), FD03 (9,399).
