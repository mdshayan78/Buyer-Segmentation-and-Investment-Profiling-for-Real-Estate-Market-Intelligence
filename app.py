import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    page_icon="🏠",
    layout="wide"
)

# Load Data
df = pd.read_csv("buyer_segmentation.csv")

st.title("🏠 Buyer Segmentation & Investment Profiling Dashboard")
st.markdown("### Machine Learning Based Real Estate Market Intelligence")

# ---------------- Sidebar ----------------
st.sidebar.header("Filters")

country = st.sidebar.selectbox(
    "Country",
    ["All"] + sorted(df["country"].dropna().unique().tolist())
)

client = st.sidebar.selectbox(
    "Client Type",
    ["All"] + sorted(df["client_type"].dropna().unique().tolist())
)

purpose = st.sidebar.selectbox(
    "Purpose",
    ["All"] + sorted(df["acquisition_purpose"].dropna().unique().tolist())
)

filtered_df = df.copy()

if country != "All":
    filtered_df = filtered_df[filtered_df["country"] == country]

if client != "All":
    filtered_df = filtered_df[filtered_df["client_type"] == client]

if purpose != "All":
    filtered_df = filtered_df[filtered_df["acquisition_purpose"] == purpose]

# ---------------- KPI ----------------
c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Buyers", len(filtered_df))
c2.metric("Countries", filtered_df["country"].nunique())
c3.metric("Avg Sale Price", f"${filtered_df['sale_price'].mean():,.0f}")
c4.metric("Avg Satisfaction", round(filtered_df["satisfaction_score"].mean(),2))

st.divider()

# ---------------- Cluster Distribution ----------------

st.subheader("Buyer Segments")

cluster = filtered_df["Cluster"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(6,4))
ax.bar(cluster.index.astype(str), cluster.values)
ax.set_xlabel("Cluster")
ax.set_ylabel("Buyers")

st.pyplot(fig)

# ---------------- Client Type ----------------

st.subheader("Client Type Distribution")

fig, ax = plt.subplots(figsize=(6,4))

filtered_df["client_type"].value_counts().plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

# ---------------- Loan ----------------

st.subheader("Loan Applied")

fig, ax = plt.subplots(figsize=(6,4))

filtered_df["loan_applied"].value_counts().plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

# ---------------- Purpose ----------------

st.subheader("Acquisition Purpose")

fig, ax = plt.subplots(figsize=(6,4))

filtered_df["acquisition_purpose"].value_counts().plot(
    kind="bar",
    ax=ax
)

st.pyplot(fig)

# ---------------- Region ----------------

st.subheader("Top Regions")

st.bar_chart(filtered_df["region"].value_counts().head(10))

# ---------------- Data ----------------

st.subheader("Filtered Dataset")

st.dataframe(filtered_df)

# ---------------- Summary ----------------

st.subheader("Cluster Summary")

st.dataframe(
    filtered_df.groupby("Cluster")[
        ["sale_price","floor_area_sqft","Age","satisfaction_score"]
    ].mean().round(2)
)
