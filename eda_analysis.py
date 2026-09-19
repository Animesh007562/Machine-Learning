"""
EDA Project: India NCRB Crime Data (2014)
==========================================
Performs exploratory data analysis on the All-State NCRB crime dataset
(state/district level, IPC crime counts for the year 2014) and produces
a set of publication-ready visualizations.

Run:
    python eda_analysis.py

Outputs:
    - Console summary statistics
    - PNG charts saved under ./plots/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------
# 0. Setup
# ----------------------------------------------------------------------
DATA_PATH = "All_State_NCRB_Crime_Data_2014.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="mako")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titleweight"] = "bold"

# Crime categories most useful for headline analysis
KEY_CRIMES = [
    "Murder", "Rape", "Kidnapping & Abduction_Total", "Dacoity",
    "Robbery", "Theft", "Riots", "Cheating", "Arson",
    "Dowry Deaths", "Total Cognizable IPC crimes",
]


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw file and splits it into:
      - district-level rows (one row per actual district)
      - state-level rows (the pre-aggregated 'Total' row NCRB includes
        per state — this equals the sum of that state's districts, so
        it must NOT be summed together with the district rows or every
        state total gets double-counted).
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    is_total_row = df["District"].str.strip().eq("Total")
    district_df = df[~is_total_row].reset_index(drop=True)
    state_df = df[is_total_row].reset_index(drop=True)

    return df, district_df, state_df


# ----------------------------------------------------------------------
# 1. Basic EDA
# ----------------------------------------------------------------------
def basic_eda(raw_df: pd.DataFrame, district_df: pd.DataFrame, state_df: pd.DataFrame) -> None:
    print("=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Raw rows (incl. per-state 'Total' rows): {raw_df.shape[0]}")
    print(f"Actual district-level rows              : {district_df.shape[0]}")
    print(f"States/UTs covered                       : {raw_df['States/UTs'].nunique()}")
    print(f"Year(s) present                          : {sorted(raw_df['Year'].unique())}")
    print(f"Missing values                            : {int(raw_df.isnull().sum().sum())}")
    print(f"Duplicate rows                            : {int(district_df.duplicated().sum())}")
    print("Note: NCRB includes one 'Total' row per state that equals the "
          "sum of its districts — these are separated out (state_df) so "
          "state-level totals aren't double-counted.")

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS — key crime categories (district-level)")
    print("=" * 70)
    print(district_df[KEY_CRIMES].describe().T.round(1))

    print("\n" + "=" * 70)
    print("TOP 10 STATES BY TOTAL COGNIZABLE IPC CRIMES")
    print("=" * 70)
    state_totals = (
        state_df.set_index("States/UTs")["Total Cognizable IPC crimes"]
        .sort_values(ascending=False)
    )
    print(state_totals.head(10))

    print("\n" + "=" * 70)
    print("TOP 10 DISTRICTS BY TOTAL COGNIZABLE IPC CRIMES")
    print("=" * 70)
    district_totals = (
        district_df[["States/UTs", "District", "Total Cognizable IPC crimes"]]
        .sort_values("Total Cognizable IPC crimes", ascending=False)
        .head(10)
    )
    print(district_totals.to_string(index=False))


# ----------------------------------------------------------------------
# 2. Visualizations
# ----------------------------------------------------------------------
def plot_top_states(state_df: pd.DataFrame, n: int = 15) -> None:
    state_totals = (
        state_df.set_index("States/UTs")["Total Cognizable IPC crimes"]
        .sort_values(ascending=False)
        .head(n)
    )
    plt.figure(figsize=(10, 7))
    sns.barplot(x=state_totals.values, y=state_totals.index, hue=state_totals.index,
                legend=False, palette="mako")
    plt.title(f"Top {n} States/UTs by Total Cognizable IPC Crimes (2014)")
    plt.xlabel("Total IPC Crimes")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_top_states_total_crime.png")
    plt.close()


def plot_top_districts(district_df: pd.DataFrame, n: int = 15) -> None:
    top = district_df.nlargest(n, "Total Cognizable IPC crimes").copy()
    top["label"] = top["District"] + " (" + top["States/UTs"] + ")"
    plt.figure(figsize=(10, 7))
    sns.barplot(x="Total Cognizable IPC crimes", y="label", data=top,
                hue="label", legend=False, palette="rocket")
    plt.title(f"Top {n} Districts by Total Cognizable IPC Crimes (2014)")
    plt.xlabel("Total IPC Crimes")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_top_districts_total_crime.png")
    plt.close()


def plot_crime_category_share(district_df: pd.DataFrame) -> None:
    cats = ["Murder", "Rape", "Kidnapping & Abduction_Total", "Dacoity",
            "Robbery", "Theft", "Riots", "Cheating", "Arson", "Dowry Deaths"]
    totals = district_df[cats].sum().sort_values(ascending=False)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=totals.values, y=totals.index, hue=totals.index,
                legend=False, palette="flare")
    plt.title("All-India Totals by Crime Category (2014)")
    plt.xlabel("Reported Cases")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_crime_category_totals.png")
    plt.close()


def plot_correlation_heatmap(district_df: pd.DataFrame) -> None:
    corr = district_df[KEY_CRIMES].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Between Key Crime Categories")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_correlation_heatmap.png")
    plt.close()


def plot_distribution(district_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(district_df["Total Cognizable IPC crimes"], bins=40, kde=True,
                 ax=axes[0], color="#3b6ba5")
    axes[0].set_title("Distribution: Total IPC Crimes per District")
    axes[0].set_xlabel("Total IPC Crimes")

    sns.boxplot(x=district_df["Murder"], ax=axes[1], color="#c1666b")
    axes[1].set_title("Spread of Murder Counts per District (outliers visible)")
    axes[1].set_xlabel("Murder Cases")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_distributions.png")
    plt.close()


def plot_murder_vs_rape(district_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=district_df, x="Murder", y="Rape", hue="States/UTs",
                     legend=False, alpha=0.6, s=40, palette="viridis")
    plt.title("Murder vs Rape Cases per District")
    plt.xlabel("Murder Cases")
    plt.ylabel("Rape Cases")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_murder_vs_rape_scatter.png")
    plt.close()


def plot_stacked_category_by_state(state_df: pd.DataFrame, n: int = 10) -> None:
    cats = ["Murder", "Rape", "Robbery", "Dacoity", "Riots"]
    top_states = (
        state_df.set_index("States/UTs")["Total Cognizable IPC crimes"]
        .sort_values(ascending=False).head(n).index
    )
    sub = state_df.set_index("States/UTs").loc[top_states, cats]
    sub.plot(kind="barh", stacked=True, figsize=(10, 7), colormap="tab10")
    plt.title(f"Violent/Property Crime Mix — Top {n} States")
    plt.xlabel("Reported Cases")
    plt.ylabel("")
    plt.legend(title="Crime Type", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_stacked_category_by_state.png")
    plt.close()


def run_all_plots(district_df: pd.DataFrame, state_df: pd.DataFrame) -> None:
    plot_top_states(state_df)
    plot_top_districts(district_df)
    plot_crime_category_share(district_df)
    plot_correlation_heatmap(district_df)
    plot_distribution(district_df)
    plot_murder_vs_rape(district_df)
    plot_stacked_category_by_state(state_df)
    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
if __name__ == "__main__":
    raw, district_data, state_data = load_data()
    basic_eda(raw, district_data, state_data)
    run_all_plots(district_data, state_data)
