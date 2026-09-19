"""
India NCRB Crime Data (2014) — Interactive Gradio Dashboard
=============================================================
Run:
    pip install gradio pandas matplotlib seaborn plotly
    python dashboard.py

Then open the local URL Gradio prints (usually http://127.0.0.1:7860).

The dashboard has four tabs:
    1. Overview        - national headline numbers + top states/districts
    2. State Explorer   - drill into one state's districts & crime mix
    3. Compare States   - side-by-side comparison of chosen states
    4. Raw Data          - filterable/searchable data table + CSV export
"""

import gradio as gr
import pandas as pd
import plotly.express as px

DATA_PATH = "All_State_NCRB_Crime_Data_2014.csv"

CRIME_COLS = [
    "Murder", "Attempt to commit Murder", "Rape", "Kidnapping & Abduction_Total",
    "Dacoity", "Robbery", "Theft", "Auto Theft", "Riots", "Cheating",
    "Forgery", "Arson", "Dowry Deaths", "Sexual Harassment", "Stalking",
    "Cruelty by Husband or his Relatives", "Extortion", "HumanTrafficking",
    "Total Cognizable IPC crimes",
]

# ----------------------------------------------------------------------
# Data loading — split the per-state "Total" rows from real districts
# ----------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    is_total = df["District"].str.strip().eq("Total")
    district_df = df[~is_total].reset_index(drop=True)
    state_df = df[is_total].reset_index(drop=True)
    return df, district_df, state_df


RAW_DF, DISTRICT_DF, STATE_DF = load_data()
STATE_LIST = sorted(STATE_DF["States/UTs"].unique().tolist())
CRIME_CHOICES = [c for c in CRIME_COLS if c in RAW_DF.columns]


# ----------------------------------------------------------------------
# Tab 1: Overview
# ----------------------------------------------------------------------
def overview_stats():
    total_crimes = int(STATE_DF["Total Cognizable IPC crimes"].sum())
    n_states = STATE_DF["States/UTs"].nunique()
    n_districts = DISTRICT_DF.shape[0]
    top_state = STATE_DF.loc[
        STATE_DF["Total Cognizable IPC crimes"].idxmax(), "States/UTs"
    ]
    summary = (
        f"### National Snapshot — 2014\n"
        f"- **Total Cognizable IPC crimes (all states/UTs):** {total_crimes:,}\n"
        f"- **States/UTs covered:** {n_states}\n"
        f"- **Districts covered:** {n_districts}\n"
        f"- **Highest-crime state/UT:** {top_state}\n"
    )
    return summary


def overview_top_states_chart(n=15):
    top = STATE_DF.nlargest(n, "Total Cognizable IPC crimes")
    fig = px.bar(
        top.sort_values("Total Cognizable IPC crimes"),
        x="Total Cognizable IPC crimes", y="States/UTs",
        orientation="h", title=f"Top {n} States/UTs by Total IPC Crimes",
        color="Total Cognizable IPC crimes", color_continuous_scale="Viridis",
    )
    fig.update_layout(height=500, showlegend=False)
    return fig


def overview_top_districts_chart(n=15):
    top = DISTRICT_DF.nlargest(n, "Total Cognizable IPC crimes").copy()
    top["label"] = top["District"] + " (" + top["States/UTs"] + ")"
    fig = px.bar(
        top.sort_values("Total Cognizable IPC crimes"),
        x="Total Cognizable IPC crimes", y="label",
        orientation="h", title=f"Top {n} Districts by Total IPC Crimes",
        color="Total Cognizable IPC crimes", color_continuous_scale="Magma",
    )
    fig.update_layout(height=500, showlegend=False, yaxis_title="")
    return fig


# ----------------------------------------------------------------------
# Tab 2: State Explorer
# ----------------------------------------------------------------------
def state_explorer(state: str, crime: str, top_n: int):
    sub = DISTRICT_DF[DISTRICT_DF["States/UTs"] == state]
    if sub.empty:
        return None, None, pd.DataFrame()

    top_districts = sub.nlargest(int(top_n), crime)[["District", crime]]
    bar_fig = px.bar(
        top_districts.sort_values(crime),
        x=crime, y="District", orientation="h",
        title=f"{state}: Top {top_n} Districts by {crime}",
        color=crime, color_continuous_scale="Teal",
    )
    bar_fig.update_layout(height=450, showlegend=False)

    mix_cols = ["Murder", "Rape", "Robbery", "Dacoity", "Riots", "Theft"]
    mix_cols = [c for c in mix_cols if c in sub.columns]
    mix_totals = sub[mix_cols].sum().reset_index()
    mix_totals.columns = ["Crime Type", "Cases"]
    pie_fig = px.pie(
        mix_totals, names="Crime Type", values="Cases",
        title=f"{state}: Crime Category Mix",
        hole=0.4,
    )
    pie_fig.update_layout(height=450)

    table = sub[["District", "Murder", "Rape", "Robbery", "Theft",
                 "Total Cognizable IPC crimes"]].sort_values(
        "Total Cognizable IPC crimes", ascending=False
    )
    return bar_fig, pie_fig, table


# ----------------------------------------------------------------------
# Tab 3: Compare States
# ----------------------------------------------------------------------
def compare_states(states: list, crimes: list):
    if not states or not crimes:
        return None, pd.DataFrame()

    sub = STATE_DF[STATE_DF["States/UTs"].isin(states)][["States/UTs"] + crimes]
    melted = sub.melt(id_vars="States/UTs", var_name="Crime Type", value_name="Cases")
    fig = px.bar(
        melted, x="States/UTs", y="Cases", color="Crime Type",
        barmode="group", title="State Comparison by Selected Crime Types",
    )
    fig.update_layout(height=500, xaxis_title="", legend_title="Crime Type")
    return fig, sub.reset_index(drop=True)


# ----------------------------------------------------------------------
# Tab 4: Raw Data explorer
# ----------------------------------------------------------------------
def filter_raw_data(state: str, search: str, min_total: int):
    df = DISTRICT_DF.copy()
    if state and state != "All":
        df = df[df["States/UTs"] == state]
    if search:
        df = df[df["District"].str.contains(search, case=False, na=False)]
    if min_total:
        df = df[df["Total Cognizable IPC crimes"] >= min_total]
    cols = ["States/UTs", "District"] + CRIME_CHOICES
    return df[cols].sort_values("Total Cognizable IPC crimes", ascending=False)


def export_csv(state: str, search: str, min_total: int):
    df = filter_raw_data(state, search, min_total)
    path = "/tmp/filtered_crime_data.csv"
    df.to_csv(path, index=False)
    return path


# ----------------------------------------------------------------------
# Build the Gradio app
# ----------------------------------------------------------------------
with gr.Blocks(title="India NCRB Crime Dashboard (2014)", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇮🇳 India NCRB Crime Data Dashboard — 2014")
    gr.Markdown(
        "State & district-level Indian Penal Code (IPC) crime statistics "
        "from the National Crime Records Bureau (NCRB), 2014."
    )

    with gr.Tab("📊 Overview"):
        stats_md = gr.Markdown(overview_stats())
        with gr.Row():
            n_states_slider = gr.Slider(5, 36, value=15, step=1, label="Number of states to show")
            n_districts_slider = gr.Slider(5, 30, value=15, step=1, label="Number of districts to show")
        with gr.Row():
            states_plot = gr.Plot(value=overview_top_states_chart(15))
            districts_plot = gr.Plot(value=overview_top_districts_chart(15))

        n_states_slider.change(overview_top_states_chart, inputs=n_states_slider, outputs=states_plot)
        n_districts_slider.change(overview_top_districts_chart, inputs=n_districts_slider, outputs=districts_plot)

    with gr.Tab("🔍 State Explorer"):
        with gr.Row():
            state_dd = gr.Dropdown(STATE_LIST, value=STATE_LIST[0], label="Select State/UT")
            crime_dd = gr.Dropdown(CRIME_CHOICES, value="Total Cognizable IPC crimes", label="Crime Type")
            top_n_slider = gr.Slider(5, 30, value=10, step=1, label="Top N Districts")
        with gr.Row():
            state_bar = gr.Plot()
            state_pie = gr.Plot()
        state_table = gr.Dataframe(label="District-level detail", wrap=True)

        explore_inputs = [state_dd, crime_dd, top_n_slider]
        explore_outputs = [state_bar, state_pie, state_table]
        for comp in explore_inputs:
            comp.change(state_explorer, inputs=explore_inputs, outputs=explore_outputs)
        demo.load(state_explorer, inputs=explore_inputs, outputs=explore_outputs)

    with gr.Tab("⚖️ Compare States"):
        with gr.Row():
            compare_states_dd = gr.Dropdown(
                STATE_LIST, value=STATE_LIST[:4], multiselect=True, label="Select States/UTs to compare"
            )
            compare_crimes_dd = gr.Dropdown(
                CRIME_CHOICES, value=["Murder", "Rape", "Theft"], multiselect=True, label="Select Crime Types"
            )
        compare_btn = gr.Button("Compare", variant="primary")
        compare_plot = gr.Plot()
        compare_table = gr.Dataframe(label="Comparison table")
        compare_btn.click(
            compare_states, inputs=[compare_states_dd, compare_crimes_dd],
            outputs=[compare_plot, compare_table],
        )

    with gr.Tab("📋 Raw Data"):
        with gr.Row():
            filter_state_dd = gr.Dropdown(["All"] + STATE_LIST, value="All", label="Filter by State/UT")
            search_box = gr.Textbox(label="Search district name")
            min_total_num = gr.Number(value=0, label="Minimum Total IPC Crimes")
        filter_btn = gr.Button("Apply Filters", variant="primary")
        raw_table = gr.Dataframe(label="Filtered data", wrap=True)
        download_btn = gr.DownloadButton("⬇️ Download filtered CSV")

        filter_inputs = [filter_state_dd, search_box, min_total_num]
        filter_btn.click(filter_raw_data, inputs=filter_inputs, outputs=raw_table)
        download_btn.click(export_csv, inputs=filter_inputs, outputs=download_btn)
        demo.load(filter_raw_data, inputs=filter_inputs, outputs=raw_table)

    gr.Markdown("---\n*Data source: NCRB (National Crime Records Bureau), 2014 IPC crime statistics.*")


if __name__ == "__main__":
    demo.launch()
