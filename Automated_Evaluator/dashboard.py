import os
import streamlit as st
import pandas as pd

st.title("📊 Experiment Comparison Dashboard")

summary_path = "results/summary.csv"

if not os.path.exists(summary_path):
	st.warning(f"Summary file not found: {summary_path}. Run evaluations to generate results.")
else:
	df = pd.read_csv(summary_path)
	if df.empty:
		st.info("No experiments found in summary (empty file)")
	else:
		st.dataframe(df)
		# If id column exists, use it as index for charts
		idx = "id" if "id" in df.columns else df.columns[0]
		numeric_cols = [c for c in ["RMSE", "MAE"] if c in df.columns]
		if numeric_cols:
			st.bar_chart(df.set_index(idx)[numeric_cols])
		else:
			st.info("No numeric metrics (RMSE/MAE) found in summary to chart.")