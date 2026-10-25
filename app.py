import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pulp
import folium
import streamlit as st
from streamlit_folium import st_folium

AREA_SQKM = 400
SEED = 46
MARGIN = 60        # Rs per order
COST_PER_KM = 3    # Rs per km (per order, one way)

ORIGIN_LAT, ORIGIN_LON = 12.94, 80.10
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.0 * np.cos(np.radians(ORIGIN_LAT))


@st.cache_data
def generate_data(area_sqkm, n_customers, n_stores, seed=SEED):
    rng = np.random.default_rng(seed)
    side_km = np.sqrt(area_sqkm)

    customers = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "x_km": rng.uniform(0, side_km, n_customers),
        "y_km": rng.uniform(0, side_km, n_customers),
        "demand": rng.uniform(0.4, 1.0, n_customers),        # orders/day
    })

    stores = pd.DataFrame({
        "store_id": np.arange(n_stores),
        "x_km": rng.uniform(0, side_km, n_stores),
        "y_km": rng.uniform(0, side_km, n_stores),
        "fixed_cost": rng.uniform(10000, 15000, n_stores),   # Rs/day
    })
    return customers, stores, side_km


def distance_matrix(customers, stores):
    dx = stores["x_km"].values[:, None] - customers["x_km"].values[None, :]
    dy = stores["y_km"].values[:, None] - customers["y_km"].values[None, :]
    return np.sqrt(dx**2 + dy**2)   # shape: (n_stores, n_customers)


def solve_model(customers, stores, D, D_MAX, P, MARGIN, COST_PER_KM):
    prob = pulp.LpProblem("Maximize_profit", pulp.LpMaximize)

    allowed_pairs = [(int(i), int(j)) for i, j in np.argwhere(D <= D_MAX)] # Allowed pairs: distance constraint

    y = prob.add_variable_dicts("y", stores.index, cat="Binary") # y[i] = 1 if dark store i is opened
    x = prob.add_variable_dicts("x", allowed_pairs, lowBound=0, upBound=1, cat="Continuous") # x[(i, j)] = 1 if customer j is served by store i

    q = customers["demand"].values      # demand of each customer
    f = stores["fixed_cost"].values     # fixed cost of each store

    # Objective fn
    prob += (pulp.lpSum(q[j] * (MARGIN - 2 * COST_PER_KM * D[i, j]) * x[(i, j)] for i, j in allowed_pairs)
             - pulp.lpSum(f[i] * y[i] for i in stores.index))

    # which stores are within D_MAX of each customer
    stores_for_customer = {}
    for i, j in allowed_pairs:
        if j not in stores_for_customer:
            stores_for_customer[j] = []
        stores_for_customer[j].append(i)

    #Constraints

    for j, store_list in stores_for_customer.items():
        prob += pulp.lpSum(x[(i, j)] for i in store_list) <= 1 # Each customer is served by at most one store

    for i, j in allowed_pairs:
        prob += x[(i, j)] <= y[i] # A customer can only be served by an open store

    prob += pulp.lpSum(y[i] for i in stores.index) <= P # At most P stores open

    result = prob.solve(pulp.HiGHS(msg=False, timeLimit=300))

    opened = [i for i in stores.index if y[i].varValue > 0.5]
    assign = {j: i for (i, j) in allowed_pairs if x[(i, j)].varValue > 0.5}
    return {"status": result.status_str, "profit": pulp.value(prob.objective), "opened": opened, "assign": assign}


def compute_kpis(sol, customers, stores, D, MARGIN, COST_PER_KM):
    q = customers["demand"].values
    f = stores["fixed_cost"].values
    served = np.array(list(sol["assign"].keys()), dtype=int)
    assigned_store = np.array(list(sol["assign"].values()), dtype=int)
    dist = D[assigned_store, served]
    orders = q[served]

    gross_margin = (orders * MARGIN).sum()
    delivery_cost = (orders * 2 * COST_PER_KM * dist).sum()
    fixed_cost = f[sol["opened"]].sum()

    kpis = {
        "Total profit (Rs/day)": round(sol["profit"]),
        "Stores opened": len(sol["opened"]),
        "Customers served": len(served),
        "Customers served (%)": round(100 * len(served) / len(customers), 1),
        "Orders served (per day)": round(orders.sum()),
        "Avg delivery distance (km)": round(dist.mean(), 2) if len(dist) else 0,
        "Max delivery distance (km)": round(dist.max(), 2) if len(dist) else 0,
        "Gross margin (Rs/day)": round(gross_margin),
        "Delivery cost (Rs/day)": round(delivery_cost),
        "Fixed cost (Rs/day)": round(fixed_cost),
    }
    return pd.DataFrame(list(kpis.items()), columns=["KPI", "Value"], dtype=object)


def km_to_latlon(x_km, y_km):
    lat = ORIGIN_LAT + y_km / KM_PER_DEG_LAT
    lon = ORIGIN_LON + x_km / KM_PER_DEG_LON
    return lat, lon


def build_map(sol, customers, stores, D_MAX, n_show=2000, seed=0):
    c_lat, c_lon = km_to_latlon(customers["x_km"].values, customers["y_km"].values)
    s_lat, s_lon = km_to_latlon(stores["x_km"].values, stores["y_km"].values)

    m = folium.Map(location=[c_lat.mean(), c_lon.mean()], zoom_start=12, tiles="Esri.WorldStreetMap")

    palette = [mcolors.to_hex(c) for c in plt.get_cmap("tab20").colors]
    store_color = {s: palette[k % len(palette)] for k, s in enumerate(sol["opened"])}

    rng = np.random.default_rng(seed)
    show = rng.choice(len(customers), size=min(n_show, len(customers)), replace=False)
    for j in show:
        s = sol["assign"].get(j)
        color = store_color[s] if s is not None else "#9e9e9e"
        folium.CircleMarker([c_lat[j], c_lon[j]], radius=2, color=color, fill=True,
                            fill_opacity=0.8, weight=0).add_to(m)

    for i in stores.index:
        if i in store_color:
            folium.Circle([s_lat[i], s_lon[i]], radius=D_MAX * 1000, color=store_color[i],
                          fill=False, weight=1, dash_array="4").add_to(m)
            n_cust = sum(1 for s in sol["assign"].values() if s == i)
            folium.Marker([s_lat[i], s_lon[i]],
                          tooltip=f"Store {i} (open) - {n_cust} customers, fixed cost Rs {stores.loc[i, 'fixed_cost']:,.0f}/day",
                          icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa")).add_to(m)
        else:
            folium.CircleMarker([s_lat[i], s_lon[i]], radius=5, color="black", fill=True,
                                fill_color="white", fill_opacity=1, weight=1.5,
                                tooltip=f"Store {i} (closed)").add_to(m)
    return m


@st.cache_data
def profit_sweep(D_MAX, P_LIST=(2, 5, 10, 20, 25)):
    customers, stores, _ = generate_data(AREA_SQKM, 10000, 50)
    D = distance_matrix(customers, stores)
    sweep = []
    for p in P_LIST:
        s = solve_model(customers, stores, D, D_MAX, p, MARGIN, COST_PER_KM)
        sweep.append({"Max stores allowed (P)": p, "Stores opened": len(s["opened"]), "Profit (Rs/day)": round(s["profit"])})
    return pd.DataFrame(sweep)


st.set_page_config(page_title="Dark Store Optimisation", layout="wide")
st.title("Dark Store Location Optimisation - Chennai")

n_stores = st.sidebar.number_input("Number of dark store candidates", min_value=1, max_value=200, value=50)
n_customers = st.sidebar.number_input("Number of customers", min_value=100, max_value=20000, value=10000, step=500)
D_MAX = st.sidebar.number_input("Max distance between store and customer (km)", min_value=0.5, max_value=30.0, value=3.0, step=0.5)
P = st.sidebar.number_input("Max stores allowed to open", min_value=1, max_value=200, value=20)

tab1, tab2 = st.tabs(["Optimisation", "Profit vs P"])

with tab1:
    if st.sidebar.button("Run optimisation"):
        customers, stores, _ = generate_data(AREA_SQKM, n_customers, n_stores)
        D = distance_matrix(customers, stores)
        sol = solve_model(customers, stores, D, D_MAX, P, MARGIN, COST_PER_KM)
        st.session_state.result = (customers, stores, D, sol, D_MAX)

    if "result" in st.session_state:
        customers, stores, D, sol, d_max = st.session_state.result
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("KPIs")
            st.dataframe(compute_kpis(sol, customers, stores, D, MARGIN, COST_PER_KM).astype(str), hide_index=True, width="stretch")
        with col2:
            st.subheader("Solution map")
            st_folium(build_map(sol, customers, stores, d_max), height=600, use_container_width=True, returned_objects=[])
    else:
        st.info("Set the inputs in the sidebar and click Run optimisation.")

with tab2:
    st.write(f"10,000 customers, 50 dark store candidates, max distance {D_MAX} km")
    if st.button("Run profit analysis"):
        st.session_state.sweep_dmax = D_MAX

    if "sweep_dmax" in st.session_state:
        sweep_df = profit_sweep(st.session_state.sweep_dmax)
        st.dataframe(sweep_df, hide_index=True)

        P_LIST = sweep_df["Max stores allowed (P)"].tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sweep_df["Max stores allowed (P)"], sweep_df["Profit (Rs/day)"], color="#2a78d6", linewidth=2, marker="o", markersize=8)
        for _, r in sweep_df.iterrows():
            ax.annotate(f"{r['Stores opened']}", (r["Max stores allowed (P)"], r["Profit (Rs/day)"]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#52514e")
        ax.set_xticks(P_LIST)
        ax.margins(x=0.08, y=0.12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.set_xlabel("Max dark stores allowed to open (P)")
        ax.set_ylabel("Profit (Rs/day)")
        ax.set_title("Profit vs max stores allowed")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
