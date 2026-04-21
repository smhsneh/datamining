import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import os

st.set_page_config(
    page_title="retail market basket analysis",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
            }

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
            }

section[data-testid="stSidebar"] {
    background-color: #161616;
    border-right: 1px solid #2a2a2a;
}

.metric-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f0c040;
}

.metric-label {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.stSlider > div > div > div { background: #f0c040 !important; }

div[data-testid="stDataFrame"] { border: 1px solid #2a2a2a; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_rules():
    path = os.path.join("data", "association_rules.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

@st.cache_data
def load_cleaned():
    path = os.path.join("data", "cleaned_retail.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]
    return df

rules_df = load_rules()
clean_df = load_cleaned()

st.markdown("## retail market basket analysis")
st.markdown(
    "<p style='color:#888; font-size:0.85rem; margin-top:-12px;'>association rule mining · apriori algorithm · uk online retail dataset</p>",
    unsafe_allow_html=True
)
st.markdown("---")

if rules_df is None or clean_df is None:
    st.error("data files not found. run preprocess.py and association_rules.py first.")
    st.stop()

with st.sidebar:
    st.markdown("### filters")

    min_support = st.slider("min support", 0.01, 0.2, 0.02, 0.005,
                            help="fraction of transactions containing the itemset")
    min_confidence = st.slider("min confidence", 0.1, 1.0, 0.3, 0.05,
                               help="how often the rule is correct")
    min_lift = st.slider("min lift", 1.0, 10.0, 1.0, 0.5,
                         help="lift > 1 means items are positively correlated")
    top_n = st.slider("top n rules to display", 5, 100, 20)

    st.markdown("---")
    search_term = st.text_input("search product in rules", placeholder="e.g. mug")

filtered = rules_df[
    (rules_df["support"] >= min_support) &
    (rules_df["confidence"] >= min_confidence) &
    (rules_df["lift"] >= min_lift)
].head(top_n)

if search_term:
    mask = (
        filtered["antecedents"].str.contains(search_term, case=False, na=False) |
        filtered["consequents"].str.contains(search_term, case=False, na=False)
    )
    filtered = filtered[mask]

col1, col2, col3, col4 = st.columns(4)

def kpi(col, value, label):
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-value'>{value}</div>
        <div class='metric-label'>{label}</div>
    </div>
    """, unsafe_allow_html=True)

kpi(col1, f"{len(filtered):,}", "rules shown")
kpi(col2, f"{rules_df['lift'].max():.2f}", "max lift")
kpi(col3, f"{clean_df['Invoice'].nunique():,}", "transactions")
kpi(col4, f"{clean_df['Description'].nunique():,}", "unique products")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["rules table", "lift vs confidence", "item network", "sales overview"])

with tab1:
    st.markdown(f"#### top {len(filtered)} association rules")
    display_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
    styled = filtered[display_cols].copy()
    styled["support"] = styled["support"].map("{:.4f}".format)
    styled["confidence"] = styled["confidence"].map("{:.4f}".format)
    styled["lift"] = styled["lift"].map("{:.2f}".format)
    st.dataframe(styled, use_container_width=True, height=420)

with tab2:
    st.markdown("#### lift vs confidence (bubble size = support)")
    fig = px.scatter(
        filtered,
        x="confidence",
        y="lift",
        size="support",
        color="lift",
        hover_data=["antecedents", "consequents", "support"],
        color_continuous_scale="YlOrRd",
        template="plotly_dark",
        labels={"confidence": "confidence", "lift": "lift"}
    )
    fig.update_layout(
        paper_bgcolor="#0e0e0e",
        plot_bgcolor="#161616",
        font_family="DM Mono",
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("#### item association network (top 30 rules)")
    net_rules = filtered.head(30)

    G = nx.DiGraph()
    for _, row in net_rules.iterrows():
        G.add_edge(row["antecedents"], row["consequents"], weight=row["lift"])

    pos = nx.spring_layout(G, seed=42, k=2)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.8, color="#444"), hoverinfo="none")

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        textfont=dict(size=8, color="#ccc"),
        hoverinfo="text",
        marker=dict(size=14, color="#f0c040", line=dict(width=1, color="#0e0e0e"))
    )

    fig2 = go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(
                         paper_bgcolor="#0e0e0e", plot_bgcolor="#0e0e0e",
                         showlegend=False, hovermode="closest",
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         margin=dict(l=20, r=20, t=20, b=20),
                         height=500
                     ))
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.markdown("#### monthly revenue trend")
    clean_df["Month"] = clean_df["InvoiceDate"].dt.to_period("M").astype(str)
    monthly = clean_df.groupby("Month")["Revenue"].sum().reset_index()

    fig3 = px.area(monthly, x="Month", y="Revenue",
                   template="plotly_dark",
                   color_discrete_sequence=["#f0c040"])
    fig3.update_layout(paper_bgcolor="#0e0e0e", plot_bgcolor="#161616",
                       font_family="DM Mono")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("#### top 15 products by revenue")
    top_rev = (
        clean_df.groupby("Description")["Revenue"]
        .sum().sort_values(ascending=False).head(15).reset_index()
    )
    fig4 = px.bar(top_rev, x="Revenue", y="Description", orientation="h",
                  template="plotly_dark", color="Revenue",
                  color_continuous_scale="YlOrRd")
    fig4.update_layout(paper_bgcolor="#0e0e0e", plot_bgcolor="#161616",
                       font_family="DM Mono", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)