import streamlit as st
import polars as pl
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from src.utils import DATA_PROCESSED

PAGE_TITLE = "Political Discourse Analysis: Brazil 2022"
PAGE_ICON = "⚖️"
DISPLAY_PATH = DATA_PROCESSED / "labeled_v4_discourse.parquet"   # text_en + labels (may be stale)
CLASS_PATH   = DATA_PROCESSED / "labeled_discourse_pt.parquet"   # labels/scores always fresh
REQUIRED_COLS = ["date", "author", "text_en", "label", "score"]
CUSTOM_STOPWORDS = set(STOPWORDS).union({"https", "t.co", "amp", "will", "brazil", "people", "president"})


def setup_page():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown("""
        <style>
        .block-container {padding-top: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.6rem;}
        .disclaimer-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 10px;
            font-size: 0.85em;
            color: #4f4f4f;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)


def _add_time_cols(df: pl.DataFrame) -> pl.DataFrame:
    rhetoric_labels   = ["retórica antidemocrática", "Antidemocratic rhetoric"]
    accusation_labels = ["acusação antidemocrática", "Antidemocratic accusation"]
    return df.with_columns([
        pl.col("date").dt.truncate("1mo").alias("month_date"),
        pl.col("date").dt.strftime("%m/%Y").alias("month_str"),
        pl.when(pl.col("label").is_in(rhetoric_labels))
          .then(pl.lit("Rhetoric"))
          .when(pl.col("label").is_in(accusation_labels))
          .then(pl.lit("Accusation"))
          .otherwise(pl.lit("Others"))
          .alias("chart_category"),
    ]).sort("date")


@st.cache_data
def load_display_data() -> pl.DataFrame:
    """Display dataset (text_en). Labels may be from an earlier inference run."""
    if not DISPLAY_PATH.exists():
        st.error(f"Critical Error: Display dataset not found at {DISPLAY_PATH}")
        return pl.DataFrame()
    return (
        _add_time_cols(pl.read_parquet(DISPLAY_PATH, columns=REQUIRED_COLS))
        .filter(pl.col("text_en").is_not_null())
    )


@st.cache_data
def load_classification_data() -> pl.DataFrame:
    """Classification dataset: always reflects the latest inference run."""
    path = CLASS_PATH if CLASS_PATH.exists() else DISPLAY_PATH
    cols = ["date", "author", "label", "score"]
    return _add_time_cols(pl.read_parquet(path, columns=cols))


def render_header():
    st.title(f"{PAGE_ICON} {PAGE_TITLE}")
    st.markdown("""
    ### Auditing Democracy: NLP Analysis of the 2022 Brazilian Presidential Election

    The 2022 Brazilian election was defined by unprecedented polarization. This project addresses a core methodological challenge: standard sentiment analysis cannot distinguish *legitimate political opposition* from *genuinely antidemocratic rhetoric*; both read as "negative," but only one represents a systemic threat to institutions.

    This pipeline classifies **6,950 posts** from the official Telegram channels of Lula and Bolsonaro (2021–2022) into **seven fine-grained categories**:

    | Category | Description |
    |---|---|
    | Retórica antidemocrática | Direct authorial attacks on democratic institutions or the electoral system |
    | Acusação antidemocrática | Attributing antidemocratic behavior to the opponent |
    | Ataques políticos | Personal, moral, or competence-based attacks on the adversary |
    | Campanha eleitoral | Electoral campaign content: promises, rallies, endorsements |
    | Administração pública | Governance updates: policy, public works, government acts |
    | Política econômica | Economic policy announcements or commentary |
    | Neutro | Informational or ceremonial content with no strong political valence |

    The classification uses **AKD (Active Knowledge Distillation)**, which combines two principles: knowledge distillation, where a *student* model is trained to approximate a larger *teacher* (Hinton et al., 2015); and active learning, which selects only the most informative samples for labeling (Luccioli et al., 2025). In practice: (1) **XLM-R** (a multilingual model that can classify text into categories without prior examples, by comparing each post against plain-language descriptions of each class) labels the full corpus and assigns a confidence score; (2) posts where the model is least confident are sent to a frontier **LLM** (`claude-opus-4-7`) acting as *teacher*, which annotates those cases; (3) the consolidated labels fine-tune **BERTimbau**, a language model pre-trained on large volumes of Brazilian Portuguese text and specialized here on the Telegram corpus. A set of **pattern-matching rules** runs before each neural prediction to resolve authorship ambiguity: words like *disse, afirmou, declarou* signal that the channel is reporting someone else's words (accusation); negations like *não, bloqueou, impediu* signal that the channel is criticizing an action (also accusation). Without this step, the phrase *"O Bolsonaro quer invadir o STF"*, despite being identical in threat-level vocabulary to direct rhetoric, would be misclassified.

    The **Antidemocratic Discourse Index (ADI)** below measures the intensity and cumulative load of hostile political discourse per candidate, separately tracking antidemocratic accusations and personal attacks, normalized by monthly volume and weighted by model confidence.
    """)
    st.info("Author: Luís G. B. Amaral (github.com/luisgbamaral)")


def render_overview(df_display: pl.DataFrame, df_class: pl.DataFrame):
    df_ref = df_class if not df_class.is_empty() else df_display
    total_posts      = df_ref.height
    count_rhetoric   = df_ref.filter(pl.col("chart_category") == "Rhetoric").height
    count_accusation = df_ref.filter(pl.col("chart_category") == "Accusation").height

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Timeline Start",    df_ref["date"].min().strftime("%d/%m/%Y"))
    c2.metric("Timeline End",      df_ref["date"].max().strftime("%d/%m/%Y"))
    c3.metric("Total Corpus",      f"{total_posts:,}")
    c4.metric("Systemic Rhetoric", f"{count_rhetoric:,}",   delta="Direct Threats",     delta_color="inverse")
    c5.metric("Accusations",       f"{count_accusation:,}", delta="Pol. Weaponization", delta_color="normal")

    st.divider()
    render_index(df_class)
    st.divider()
    render_findings(df_class, df_display)


def render_explorer(df: pl.DataFrame):
    st.header("🔎 Deep Dive Explorer")
    st.markdown("Inspect the raw data and the model's classification logic.")

    col1, col2 = st.columns(2)
    all_authors = df["author"].unique().sort().to_list()
    all_labels  = df["label"].unique().sort().to_list()

    sel_authors = col1.multiselect("Filter Author", all_authors, default=all_authors)
    sel_labels  = col2.multiselect("Filter Specific Label", all_labels, default=all_labels)

    filtered_df = df.filter(
        (pl.col("author").is_in(sel_authors if sel_authors else all_authors)) &
        (pl.col("label").is_in(sel_labels  if sel_labels  else all_labels))
    )

    st.dataframe(
        filtered_df.sort("score", descending=True).select(["date", "author", "label", "score", "text_en"]),
        column_config={
            "score": st.column_config.ProgressColumn("Confidence", format="%.2f", min_value=0, max_value=1),
            "label": st.column_config.TextColumn("Detailed Category"),
            "text_en": "Translated Content",
        },
        use_container_width=True,
        hide_index=True,
    )


def render_cloud(df: pl.DataFrame):
    st.header("☁️ Semantic Analysis")
    col1, col2 = st.columns([1, 3])

    with col1:
        target     = st.radio("Corpus Source", ["Both Candidates", "Bolsonaro", "Lula"])
        cat_filter = st.multiselect("Focus Area", ["Rhetoric", "Accusation"], default=["Rhetoric", "Accusation"])

    subset = df.filter(pl.col("chart_category").is_in(cat_filter))
    if target != "Both Candidates":
        subset = subset.filter(pl.col("author") == target)

    if subset.height == 0:
        st.warning("Insufficient data to generate word cloud.")
        return

    with col2:
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            stopwords=CUSTOM_STOPWORDS,
            colormap="Reds",
        ).generate(" ".join(subset["text_en"].to_list()))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)


@st.cache_data
def compute_adi(_df: pl.DataFrame) -> pl.DataFrame:
    df = _df
    return (
        df
        .with_columns([
            pl.when(pl.col("label") == "acusação antidemocrática")
              .then(pl.col("score")).otherwise(0.0).alias("accusation_score"),
            pl.when(pl.col("label") == "ataques políticos")
              .then(pl.col("score")).otherwise(0.0).alias("attacks_score"),
        ])
        .group_by(["month_date", "month_str", "author"])
        .agg([
            pl.len().alias("total_posts"),
            pl.col("accusation_score").sum().alias("accusation_score_sum"),
            pl.col("attacks_score").sum().alias("attacks_score_sum"),
        ])
        .with_columns([
            (pl.col("accusation_score_sum") / pl.col("total_posts") * 100).alias("accusation_monthly"),
            (pl.col("attacks_score_sum")    / pl.col("total_posts") * 100).alias("attacks_monthly"),
        ])
        .sort(["author", "month_date"])
        .with_columns([
            pl.col("accusation_monthly").cum_sum().over("author").alias("accusation_cumulative"),
            pl.col("attacks_monthly").cum_sum().over("author").alias("attacks_cumulative"),
        ])
    )


def render_index(df: pl.DataFrame):
    st.header("📊 Antidemocratic Discourse Index (ADI)")
    st.markdown(
        "For each month and each candidate, the ADI is: "
        "**sum of confidence scores of posts in the target class ÷ total posts that month × 100**. "
        "Example: if a candidate posts 100 messages in a month and 10 are classified as *ataques políticos* "
        "with average confidence 0.80, the attacks ADI for that month is 10 × 0.80 ÷ 100 × 100 = 8. "
        "Dividing by total posts removes the effect of posting volume: a candidate who tweets 10× more "
        "is not automatically rated higher; what matters is the *proportion* of posts dedicated to attacks.\n\n"
        "**Reading the chart:** The **bars** (left axis) show the score for each individual month. "
        "The **lines** (right axis) show the running cumulative total since the start of the period; "
        "the two scales are independent and should not be compared directly. "
        "Two components are tracked separately: "
        "**Antidemocratic Accusations** (attributing antidemocratic behavior to the opponent) "
        "and **Political Attacks** (personal, moral, or competence-based attacks)."
    )

    adi = compute_adi(df)
    colors = {"Bolsonaro": "#1f77b4", "Lula": "#d62728"}

    for author in ["Bolsonaro", "Lula"]:
        subset = adi.filter(pl.col("author") == author)
        if subset.is_empty():
            continue
        color = colors[author]
        acc_final = subset["accusation_cumulative"].max()
        att_final = subset["attacks_cumulative"].max()

        st.subheader(author)
        c1, c2 = st.columns(2)
        c1.metric("Accusation ADI (cumulative)", f"{acc_final:.2f}")
        c2.metric("Political Attacks ADI (cumulative)", f"{att_final:.2f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=subset["month_str"], y=subset["accusation_monthly"],
            name="Antidemocratic Accusations", marker_color=color, opacity=0.5, yaxis="y1",
        ))
        fig.add_trace(go.Bar(
            x=subset["month_str"], y=subset["attacks_monthly"],
            name="Political Attacks", marker_color=color, opacity=0.25, yaxis="y1",
        ))
        fig.add_trace(go.Scatter(
            x=subset["month_str"], y=subset["accusation_cumulative"],
            name="Accusation (cumulative)", mode="lines+markers",
            line=dict(color=color, width=2), yaxis="y2",
        ))
        fig.add_trace(go.Scatter(
            x=subset["month_str"], y=subset["attacks_cumulative"],
            name="Attacks (cumulative)", mode="lines+markers",
            line=dict(color=color, width=2, dash="dot"), yaxis="y2",
        ))
        fig.update_layout(
            yaxis=dict(title="Monthly ADI score (bars)", side="left", showgrid=False),
            yaxis2=dict(title="Cumulative ADI: running total (lines)", side="right", overlaying="y", showgrid=True),
            barmode="group", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Monthly Values")
    st.dataframe(
        adi.sort(["author", "month_date"]).select([
            "author", "month_str", "total_posts",
            "accusation_monthly", "accusation_cumulative",
            "attacks_monthly", "attacks_cumulative",
        ]),
        column_config={
            "author":                 "Candidate",
            "month_str":              "Month",
            "total_posts":            "Total Posts",
            "accusation_monthly":     st.column_config.NumberColumn("Accusation Monthly",     format="%.3f"),
            "accusation_cumulative":  st.column_config.NumberColumn("Accusation Cumulative",  format="%.3f"),
            "attacks_monthly":        st.column_config.NumberColumn("Attacks Monthly",        format="%.3f"),
            "attacks_cumulative":     st.column_config.NumberColumn("Attacks Cumulative",     format="%.3f"),
        },
        use_container_width=True,
        hide_index=True,
    )


def render_findings(df_class: pl.DataFrame, df_display: pl.DataFrame):
    st.header("📋 Research Findings")

    adi = compute_adi(df_class)

    # Finding 1
    st.subheader("Finding 1: Official Channels Do Not Employ Direct Antidemocratic Rhetoric")

    rhetoric_posts = df_display.filter(pl.col("label") == "retórica antidemocrática").sort("date")
    n_rhetoric = rhetoric_posts.height
    n_word = "post" if n_rhetoric == 1 else "posts"
    qualifies = "qualifies" if n_rhetoric == 1 else "qualify"

    if n_rhetoric == 0:
        n_label = "no posts"
        author_note = ""
    else:
        n_label = f"**{n_rhetoric} {n_word}**"
        ret_authors = rhetoric_posts["author"].unique().to_list()
        if len(ret_authors) == 1:
            both_str = "both " if n_rhetoric > 1 else ""
            author_note = f", {both_str}from {ret_authors[0]},"
        else:
            author_note = ""

    st.markdown(f"""
    The original hypothesis was that the 2022 election was heavily marked by direct antidemocratic
    rhetoric in official channels. The AKD pipeline systematically reviewed the highest-stakes
    posts (electoral system, military intervention, judicial delegitimization) and
    **consistently classified them as accusations** against the opponent, not as direct authorial
    rhetoric. Manual validation confirmed: across 6,950 messages, {n_label}{author_note} {qualifies} as direct
    authorial antidemocratic rhetoric.
    """)

    if n_rhetoric > 0:
        cols = st.columns(n_rhetoric)
        for col, row in zip(cols, rhetoric_posts.iter_rows(named=True)):
            date_str = row["date"].strftime("%b %Y")
            col.info(f"**{date_str}**\n\n*\"{row['text_en']}\"*\n\nConfidence: {row['score']:.2f}")

    st.markdown("""
    **Conclusion.** Official channels function as governance and campaign platforms managed by
    professional press offices. Direct rhetoric (live threats, military mobilization, outright
    electoral refusal) belongs to rallies and informal interactions outside this corpus.
    The key phenomenon is not direct rhetoric but **the frequency and intensity of accusations
    against the adversary**, which proved frequent and quantifiable.
    """)

    st.divider()

    # Finding 2
    st.subheader("Finding 2: Accusations Against the Opponent Are Frequent and Grow During the Electoral Period")

    bol = adi.filter(pl.col("author") == "Bolsonaro")
    lula = adi.filter(pl.col("author") == "Lula")

    bol_acc  = bol["accusation_cumulative"].max()
    bol_att  = bol["attacks_cumulative"].max()
    lula_acc = lula["accusation_cumulative"].max()
    lula_att = lula["attacks_cumulative"].max()

    bol_acc_peak  = bol.filter(pl.col("accusation_monthly") == bol["accusation_monthly"].max())["month_str"][0]
    lula_acc_peak = lula.filter(pl.col("accusation_monthly") == lula["accusation_monthly"].max())["month_str"][0]
    bol_att_peak  = bol.filter(pl.col("attacks_monthly") == bol["attacks_monthly"].max())["month_str"][0]
    lula_att_peak = lula.filter(pl.col("attacks_monthly") == lula["attacks_monthly"].max())["month_str"][0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Bolsonaro**")
        st.metric("Antidemocratic Accusations (cumulative ADI)", f"{bol_acc:.2f}")
        st.metric("Political Attacks (cumulative ADI)", f"{bol_att:.2f}")
        st.caption(f"Peak accusation month: **{bol_acc_peak}** · Peak attacks month: **{bol_att_peak}**")
    with c2:
        st.markdown("**Lula**")
        st.metric("Antidemocratic Accusations (cumulative ADI)", f"{lula_acc:.2f}")
        st.metric("Political Attacks (cumulative ADI)", f"{lula_att:.2f}")
        st.caption(f"Peak accusation month: **{lula_acc_peak}** · Peak attacks month: **{lula_att_peak}**")

    acc_leader  = "Bolsonaro" if bol_acc  > lula_acc  else "Lula"
    att_leader  = "Bolsonaro" if bol_att  > lula_att  else "Lula"
    acc_ratio   = max(bol_acc, lula_acc) / max(min(bol_acc, lula_acc), 0.001)
    att_ratio   = max(bol_att, lula_att) / max(min(bol_att, lula_att), 0.001)

    st.markdown(f"""
    **{acc_leader}** accumulated a higher load of antidemocratic accusations ({acc_ratio:.1f}× the rival).
    **{att_leader}** led in personal and competence-based attacks ({att_ratio:.1f}×).
    Peak months: Bolsonaro's accusations in **{bol_acc_peak}**, attacks in **{bol_att_peak}**;
    Lula's accusations in **{lula_acc_peak}**, attacks in **{lula_att_peak}**.

    **{att_leader}**'s channel deployed personal and competence-based attacks at a significantly
    higher rate ({att_ratio:.1f}× the rival), targeting the opponent's fitness for office.
    **Hostile political discourse in this corpus is concentrated in {att_leader}'s channel and
    expressed primarily through competence-based attacks rather than direct rhetoric.**
    This pattern is consistent with a broader dynamic observed across electoral contexts: incumbents
    tend to highlight governance achievements and policy deliverables, while challengers are more
    likely to question the adversary's record and fitness for office.
    """)



def render_footer():
    st.markdown("---")
    st.markdown("""
        <div class='disclaimer-box'>
            <strong>Research Disclaimer</strong><br><br>
            This pipeline uses <strong>AKD (Active Knowledge Distillation)</strong>: XLM-R zero-shot labels the full corpus and assigns a confidence score per post; the lowest-confidence subset is sent to a frontier LLM (<code>claude-opus-4-7</code>) acting as <em>teacher</em> for annotation; the consolidated labels fine-tune BERTimbau on the target domain. A pattern-matching rule layer runs before each neural prediction to resolve authorship ambiguity between direct rhetoric and accusations.
            <br><br>
            <em>Limitations:</em> The regex layer relies on a fixed lexicon of political terms and may not generalize beyond the 2022 Brazilian electoral context. The fine-tuned model's ceiling is bounded by the quality of the LLM annotations and the size of the labeled sample (773 examples across 7 classes).
        </div>
    """, unsafe_allow_html=True)


def main():
    setup_page()
    render_header()

    df_display = load_display_data()
    df_class   = load_classification_data()

    if df_display.is_empty() and df_class.is_empty():
        return

    st.sidebar.title("🎛️ Controls")
    min_conf = st.sidebar.slider("Min. Confidence Score", 0.0, 1.0, 0.45, 0.05,
                                 help="Filter out low-confidence predictions.")
    df_clean = df_display.filter(pl.col("score") >= min_conf)

    selection = st.sidebar.radio("Navigate", [
        "📈 Dashboard Overview",
        "💾 Data Explorer",
        "☁️ Semantic Cloud",
    ])

    if selection == "📈 Dashboard Overview":
        render_overview(df_display, df_class)
    elif selection == "💾 Data Explorer":
        render_explorer(df_clean)
    elif selection == "☁️ Semantic Cloud":
        render_cloud(df_clean)

    render_footer()


if __name__ == "__main__":
    main()
