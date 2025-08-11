# app.py
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

# ===============================
# Core Normalization Functions
# ===============================
def compute_shift_percentiles_from_totals(totals_df: pd.DataFrame,
                                          student_id_col="student_id",
                                          shift_col="shift_id",
                                          score_col="raw_score") -> pd.DataFrame:
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
    tie_val = trace['overall_percentile']
    tie_group = merged_df[merged_df['overall_percentile'] == tie_val]
    trace['tie_group_size'] = int(len(tie_group))
    if trace['tie_group_size'] > 1:
        trace['tie_resolution'] = {}
        if subject_percentiles is not None and subject_priority is not None:
            pivot = subject_percentiles.pivot(index='student_id', columns='subject_id', values='subject_percentile')
            order_keys = {}
            for sid2 in tie_group['student_id']:
                keys = []
                for subj in subject_priority:
                    try:
                        keys.append(float(pivot.at[sid2, subj]))
                    except Exception:
                        keys.append(float('-inf'))
                order_keys[sid2] = keys
            trace['tie_resolution']['method'] = 'subject_priority'
            trace['tie_resolution']['subject_priority'] = subject_priority
            trace['tie_resolution']['order_keys'] = order_keys
        elif dob_df is not None and 'date_of_birth' in dob_df.columns:
            dob_map = dob_df.set_index('student_id')['date_of_birth'].to_dict()
            trace['tie_resolution']['method'] = 'age'
            trace['tie_resolution']['dob'] = {sid2: str(dob_map.get(sid2)) for sid2 in tie_group['student_id']}
        else:
            trace['tie_resolution']['method'] = 'none_available'
    else:
        trace['tie_resolution'] = {'method': 'not_applicable'}
    row_final = merged_df[merged_df['student_id'] == sid]
    trace['final_rank'] = int(row_final.get('final_rank', row_final.index[0] + 1))
    return trace


def run_normalization_audit(totals_df, subject_percentiles=None, subject_priority=None, dob_df=None):
    sp_df = compute_shift_percentiles_from_totals(totals_df)
    merged_df = merge_percentiles_and_rank(sp_df)
    final_df = apply_subject_tiebreak_and_age(merged_df, subject_percentiles, subject_priority, dob_df)
    return sp_df, merged_df, final_df


def generate_synthetic_data():
    np.random.seed(42)
    shifts = ["A", "B", "C"]
    data = []
    for shift in shifts:
        for i in range(1, 11):
            score = np.random.randint(50, 101)
            data.append([f"{shift}_S{i}", shift, score])
    return pd.DataFrame(data, columns=["student_id", "shift_id", "raw_score"])

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="EduFair - Normalization Audit", layout="wide")
st.title("📊 EduFair – Normalization Audit Tool")

use_sample = st.sidebar.button("🎲 Generate Sample Data")
uploaded_file = st.sidebar.file_uploader("Upload Totals CSV", type=["csv"])
dob_file = st.sidebar.file_uploader("Upload DOB CSV (optional)", type=["csv"])
subject_file = st.sidebar.file_uploader("Upload Subject Percentiles CSV (optional)", type=["csv"])
subject_priority = st.sidebar.text_input("Subject Priority (comma-separated)")

if use_sample:
    totals_df = generate_synthetic_data()
    st.success("Sample data generated!")
elif uploaded_file:
    totals_df = pd.read_csv(uploaded_file)
else:
    totals_df = None

dob_df = pd.read_csv(dob_file) if dob_file else None
subject_percentiles_df = pd.read_csv(subject_file) if subject_file else None
subject_priority_list = [s.strip() for s in subject_priority.split(",")] if subject_priority else None

if totals_df is not None:
    st.subheader("Totals Data")
    st.dataframe(totals_df)

    sp_df, merged_df, final_df = run_normalization_audit(
        totals_df,
        subject_percentiles=subject_percentiles_df,
        subject_priority=subject_priority_list,
        dob_df=dob_df
    )

    st.subheader("Step 1 – Shift Percentiles")
    st.dataframe(sp_df)

    st.subheader("Step 2 – Merged Percentiles & Preliminary Ranks")
    st.dataframe(merged_df)

    st.subheader("Step 3 – Final Ranks")
    st.dataframe(final_df)

    student_choice = st.selectbox("Select Student ID to Explain", final_df['student_id'].unique())
    if st.button("Explain Candidate"):
        trace = explain_candidate_normalization(
            student_choice,
            totals_df,
            sp_df,
            final_df,
            subject_percentiles=subject_percentiles_df,
            subject_priority=subject_priority_list,
            dob_df=dob_df
        )
        st.json(trace)

    csv_data = final_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Final Ranks", csv_data, "results.csv", "text/csv")
else:
    st.info("Upload a CSV or click Generate Sample Data to start.")
