import streamlit as st
import numpy as np
import pandas as pd


# Define Fitness Function
def total_cost(Q, D, S, H):
    cost = 0
    for i in range(len(Q)):
        if Q[i] <= 0:
            return float("inf")  # Penalize negative or zero quantities
        cost += (D[i] * S[i] / Q[i]) + (Q[i] * H[i] / 2)
    return cost


st.set_page_config(
    page_title="PSO Algorithm",
    layout="wide",
)

st.title("Partical Swarm Optimization")

D, S, H = [], [], []

if "n" not in st.session_state:
    st.session_state.n = 1

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.header("Products")

with col2:
    st.header("Demand (D)")

with col3:
    st.header("Ordering Cost (S)")

with col4:
    st.header("Holding Cost (H)")

N = st.session_state.n

for i in range(N):
    with col1:
        st.text_input(
            "Product",
            value=f"Product {i+1}",
            disabled=True,
            key=f"product{i+1}",
            label_visibility="collapsed",
        )

    with col2:
        d = st.number_input(
            "Demands",
            min_value=1,
            label_visibility="collapsed",
            key=f"D{i+1}",
        )
        D.append(d)

    with col3:
        s = st.number_input(
            "Orderings",
            min_value=1,
            label_visibility="collapsed",
            key=f"O{i+1}",
        )
        S.append(s)

    with col4:
        h = st.number_input(
            "Holdings",
            min_value=1,
            label_visibility="collapsed",
            key=f"H{i+1}",
        )
        H.append(h)

col_Add, col_Remove, col_start = st.columns([1, 1.15, 9])

with col_Add:
    if st.button("ADD", key="add_objet", help="Add product"):
        st.session_state.n += 1
        st.rerun()

with col_Remove:
    if N > 1:
        if st.button("REMOVE", key="remove_product", help="Delete product"):
            st.session_state.n -= 1
            st.rerun()

with col_start:
    start = st.button("START", key="START")

# PSO Parameters
num_particles = 30
num_iterations = 10
Q_min = 10
Q_max = 1000
results = {"costs": []}

if start:
    # Initialize Particles and Velocities
    particles = np.random.uniform(Q_min, Q_max, (num_particles, N))
    velocities = np.random.uniform(-10, 10, (num_particles, N))

    # Initialize Best Positions
    pBest = particles.copy()
    pBest_cost = np.array([total_cost(p, D, S, H) for p in particles])
    gBest = pBest[np.argmin(pBest_cost)]

    # PSO Loop
    for t in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()

            # Update velocity
            velocities[i] = (
                0.5 * velocities[i]
                + 1.5 * r1 * (pBest[i] - particles[i])
                + 2.0 * r2 * (gBest - particles[i])
            )

            # Update position
            particles[i] += velocities[i]

            # Boundary check
            particles[i] = np.clip(particles[i], Q_min, Q_max)

            # Evaluate new fitness
            current_cost = total_cost(particles[i], D, S, H)

            # Update personal best
            if current_cost < pBest_cost[i]:
                pBest[i] = particles[i].copy()
                pBest_cost[i] = current_cost

        # Update global best
        gBest = pBest[np.argmin(pBest_cost)]
        for i in range(N):
            if f"Product{i}" in results:
                results[f"Product{i}"].append(gBest[i])
            else:
                results[f"Product{i}"] = [gBest[i]]

        results["costs"].append(total_cost(gBest, D, S, H))

    # Final Output
    st.write(f"Optimal Order Quantities: {gBest}")
    st.write(f"Minimum Total Cost: {total_cost(gBest, D, S, H):.2f}")

    data = pd.DataFrame(results)
    left_col, center_col, right_col = st.columns([1, 2, 1])
    with center_col:
        st.line_chart(data)
