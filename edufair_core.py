# edufair_core.py
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

# -----------------------------
# 1. Normalization Audit Functions
# -----------------------------

def compute_shift_percentiles_from_totals(totals_df: pd.DataFrame,
                                          student_id_col="student_id",
                                          shift_col="shift_id",
                                          score_col="raw_score") -> pd.DataFrame:
    """Compute shift percentile for each student within their shift."""
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
                               score_col='raw_score') -> pd.DataFrame:
    """Merge shift percentiles and compute preliminary ranking."""
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
    """Apply subject-priority or age-based tie-break rules."""
    df = df_prelim.copy()
    pivot = None
    if subject_percentiles is not None and subject_priority:
        pivot = subject_percentiles.pivot(index=student_id_col, columns='subject_id', values='subject_percentile')

    if dob_df is not None:
        df = df.merge(dob_df[[student_id_col, 'date_of_birth']], on=student_id_col, how='left')

    final_order = []
    for _, group in df.groupby('overall_percentile', sort=False):
        if len(group) == 1:
            final_order.append(group)
            continue
        if pivot is not None:
            def key_fn(row):
                sid = row[student_id_col]
                keys = []
                for subj in subject_priority:
                    try:
                        keys.append(float(pivot.at[sid, subj]))
                    except Exception:
                        keys.append(float('-inf'))
                return tuple([-k for k in keys])
            rows_sorted = sorted(group.to_dict('records'), key=key_fn)
            group_sorted = pd.DataFrame(rows_sorted)
        elif 'date_of_birth' in group.columns:
            group_sorted = group.sort_values(['date_of_birth'], ascending=True)
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
    """Generate explainable trace for a single student's normalization process."""
    trace = {}
    srow = totals_df[totals_df['student_id'] == student_id].iloc[0]
    sid = student_id
    shift = srow['shift_id']
    raw = float(srow['raw_score'])
    trace['student_id'] = sid
    trace['shift_id'] = shift
    trace['raw_score'] = raw

    sh = shift_percentiles_df[shift_percentiles_df['shift_id'] == shift]
    trace['shift_n_candidates'] = int(len(sh))
    trace['shift_max_score'] = float(sh['raw_score'].max())
    trace['shift_min_score'] = float(sh['raw_score'].min())

    sp = shift_percentiles_df[(shift_percentiles_df['student_id'] == sid)]
    trace['shift_percentile'] = float(sp['shift_percentile'].iloc[0])

    merged_row = merged_df[merged_df['student_id'] == sid]
    trace['overall_percentile'] = float(merged_row['overall_percentile'].iloc[0])
    trace['prelim_rank'] = int(merged_row.get('prelim_rank', merged_row.index[0] + 1))

    tie_group = merged_df[merged_df['overall_percentile'] == trace['overall_percentile']]
    trace['tie_group_size'] = int(len(tie_group))
    if trace['tie_group_size'] > 1:
        trace['tie_resolution'] = {'method': 'tie_break_applied'}
    else:
        trace['tie_resolution'] = {'method': 'not_applicable'}

    if 'final_rank' in merged_row.columns:
        trace['final_rank'] = int(merged_row['final_rank'].iloc[0])
    else:
        trace['final_rank'] = int(merged_row.index[0] + 1)
    return trace

# -----------------------------
# 2. Synthetic Data Generator
# -----------------------------
def generate_synthetic_data(n_students=30, n_shifts=3, seed=42):
    """Generate fake exam scores for quick demo/testing."""
    np.random.seed(seed)
    students = []
    for shift in range(1, n_shifts + 1):
        for i in range(n_students // n_shifts):
            sid = f"S{shift}{i+1}"
            score = np.random.randint(50, 100)
            students.append((sid, f"Shift_{shift}", score))
    return pd.DataFrame(students, columns=["student_id", "shift_id", "raw_score"])

# -----------------------------
# 3. Main Wrapper
# -----------------------------
def run_normalization_audit(totals_df: pd.DataFrame,
                            subject_percentiles: Optional[pd.DataFrame] = None,
                            subject_priority: Optional[List[str]] = None,
                            dob_df: Optional[pd.DataFrame] = None):
    """Full normalization process in one function."""
    shift_percentiles_df = compute_shift_percentiles_from_totals(totals_df)
    merged_df = merge_percentiles_and_rank(shift_percentiles_df)
    final_df = apply_subject_tiebreak_and_age(merged_df, subject_percentiles, subject_priority, dob_df)
    return shift_percentiles_df, merged_df, final_df

if __name__ == "__main__":
    # Quick test
    df = generate_synthetic_data()
    sp_df, merged, final = run_normalization_audit(df)
    print(final)
    print(explain_candidate_normalization(final['student_id'].iloc[0], df, sp_df, final))
