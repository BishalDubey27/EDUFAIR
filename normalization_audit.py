import pandas as pd
import numpy as np
from typing import List, Optional, Dict


def compute_shift_percentiles_from_totals(totals_df: pd.DataFrame,
                                          student_id_col="student_id",
                                          shift_col="shift_id",
                                          score_col="raw_score") -> pd.DataFrame:
    """
    Given a totals_df with one row per student (student_id, shift_id, raw_score),
    compute shift_percentile as (# <= score)/N * 100 rounded to 7 decimals.
    """
    def proc(df):
        df_sorted = df.sort_values(score_col, ascending=True).reset_index(drop=True)
        n = len(df_sorted)
        df_sorted['count_le'] = df_sorted[score_col].rank(method='max')
        df_sorted['shift_percentile'] = (df_sorted['count_le'] / n) * 100.0
        df_sorted['shift_percentile'] = df_sorted['shift_percentile'].round(7)
        return df_sorted[[student_id_col, shift_col, score_col, 'shift_percentile']]

    return totals_df.groupby(shift_col, group_keys=False).apply(proc).reset_index(drop=True)


def merge_percentiles_and_rank(shift_percentiles_df: pd.DataFrame,
                               student_id_col='student_id',
                               shift_col='shift_id',
                               score_col='raw_score') -> pd.DataFrame:
    """
    Merge shift percentiles across shifts and compute overall ranking.
    """
    df = shift_percentiles_df.copy()
    df['overall_percentile'] = df['shift_percentile']
    df = df.sort_values(['overall_percentile', score_col], ascending=[False, False]).reset_index(drop=True)
    df['prelim_rank'] = df['overall_percentile'].rank(method='dense', ascending=False).astype(int)
    return df


def apply_subject_tiebreak_and_age(df_prelim: pd.DataFrame,
                                   subject_percentiles: Optional[pd.DataFrame] = None,
                                   subject_priority: Optional[List[str]] = None,
                                   dob_df: Optional[pd.DataFrame] = None,
                                   student_id_col='student_id') -> pd.DataFrame:
    """
    Apply subject-priority or age-based tie-break rules.
    """
    df = df_prelim.copy()

    # Pivot subject percentiles if provided
    pivot = None
    if subject_percentiles is not None and subject_priority:
        pivot = subject_percentiles.pivot(index=student_id_col, columns='subject_id', values='subject_percentile')

    # Merge DOB if provided
    if dob_df is not None:
        df = df.merge(dob_df[[student_id_col, 'date_of_birth']], on=student_id_col, how='left')

    final_order = []
    for _, group in df.groupby('overall_percentile', sort=False):
        if len(group) == 1:
            final_order.append(group)
            continue

        if pivot is not None:
            # Subject priority sorting
            def key_fn(row):
                sid = row[student_id_col]
                return tuple([-float(pivot.at[sid, subj]) if subj in pivot.columns else float('inf')
                              for subj in subject_priority])
            rows_sorted = sorted(group.to_dict('records'), key=key_fn)
            group_sorted = pd.DataFrame(rows_sorted)
        elif 'date_of_birth' in group.columns:
            group_sorted = group.sort_values(['date_of_birth'], ascending=True)  # Older first
        else:
            group_sorted = group.sort_values(['raw_score'], ascending=False)

        final_order.append(group_sorted)

    df_final = pd.concat(final_order, ignore_index=True)
    df_final['final_rank'] = df_final.index + 1
    return df_final


def explain_candidate_normalization(student_id: str,
                                    totals_df: pd.DataFrame,
                                    shift_percentiles_df: pd.DataFrame,
                                    merged_df: pd.DataFrame,
                                    subject_percentiles: Optional[pd.DataFrame] = None,
                                    subject_priority: Optional[List[str]] = None,
                                    dob_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Produce an explainable trace for a single student's normalization journey.
    """
    trace = {}
    srow = totals_df[totals_df['student_id'] == student_id].iloc[0]
    shift = srow['shift_id']
    raw = float(srow['raw_score'])

    trace['student_id'] = student_id
    trace['shift_id'] = shift
    trace['raw_score'] = raw

    # Shift stats
    sh = shift_percentiles_df[shift_percentiles_df['shift_id'] == shift]
    trace['shift_n_candidates'] = len(sh)
    trace['shift_max_score'] = sh['raw_score'].max()
    trace['shift_min_score'] = sh['raw_score'].min()

    # Student's shift percentile
    sp = shift_percentiles_df[(shift_percentiles_df['student_id'] == student_id)]
    trace['shift_percentile'] = float(sp['shift_percentile'].iloc[0])

    # Overall percentile & prelim rank
    merged_row = merged_df[merged_df['student_id'] == student_id]
    trace['overall_percentile'] = float(merged_row['overall_percentile'].iloc[0])
    trace['prelim_rank'] = int(merged_row['prelim_rank'].iloc[0])

    # Tie-break info
    tie_group = merged_df[merged_df['overall_percentile'] == trace['overall_percentile']]
    trace['tie_group_size'] = len(tie_group)

    if trace['tie_group_size'] > 1:
        trace['tie_resolution'] = {}
        if subject_percentiles is not None and subject_priority is not None:
            pivot = subject_percentiles.pivot(index='student_id', columns='subject_id', values='subject_percentile')
            order_keys = {
                sid: [pivot.at[sid, subj] if subj in pivot.columns else None for subj in subject_priority]
                for sid in tie_group['student_id']
            }
            trace['tie_resolution']['method'] = 'subject_priority'
            trace['tie_resolution']['order_keys'] = order_keys
        elif dob_df is not None:
            dob_map = dob_df.set_index('student_id')['date_of_birth'].to_dict()
            trace['tie_resolution']['method'] = 'age'
            trace['tie_resolution']['dob'] = {sid: str(dob_map.get(sid)) for sid in tie_group['student_id']}
        else:
            trace['tie_resolution']['method'] = 'none_available'
    else:
        trace['tie_resolution'] = {'method': 'not_applicable'}

    # Final rank
    trace['final_rank'] = int(merged_row['final_rank'].iloc[0]) if 'final_rank' in merged_row.columns else None
    return trace


if __name__ == "__main__":
    # Test run with synthetic data
    totals_df = pd.DataFrame({
        "student_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "shift_id": ["A", "A", "A", "B", "B", "B"],
        "raw_score": [80, 60, 90, 85, 70, 95]
    })

    print("\n--- Step 1: Shift Percentiles ---")
    shift_percentiles_df = compute_shift_percentiles_from_totals(totals_df)
    print(shift_percentiles_df)

    print("\n--- Step 2: Merge Percentiles ---")
    merged_df = merge_percentiles_and_rank(shift_percentiles_df)
    print(merged_df)

    print("\n--- Step 3: Apply Tie-breaks (none here) ---")
    final_df = apply_subject_tiebreak_and_age(merged_df)
    print(final_df)

    print("\n--- Step 4: Explain Candidate S1 ---")
    from pprint import pprint
    pprint(explain_candidate_normalization("S1", totals_df, shift_percentiles_df, final_df))
