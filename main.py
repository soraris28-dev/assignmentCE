import streamlit as st
import csv
import random
import pandas as pd
import io

# -------------------- Minimal UI tweaks (hide Streamlit chrome) --------------------
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Page Title --------------------
st.title("Genetic Algorithm TV Scheduler")

# -------------------- Helper: Read CSV (flexible) --------------------
def parse_csv_text(text):
    """
    Parse CSV text and return a mapping: program -> list of 18 ratings (06:00..23:00).
    Supports two common formats:
      1) Wide format: Program, r06, r07, ..., r23  (18 rating columns)
      2) Long format: program, hour, rating (multiple rows per program)
    """
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        return {}

    header_lc = [h.strip().lower() for h in header]

    # prepare result dict (program -> 18-list)
    results = {}

    # detect long format: must have 'program' (or similar), 'hour' and 'rating' columns
    if any(h in header_lc for h in ["program", "rancangan", "name"]) and "hour" in header_lc and "rating" in header_lc:
        # find indices
        # prefer 'program' else fallbacks
        prog_candidates = ["program", "rancangan", "name"]
        prog_idx = next((i for i,h in enumerate(header_lc) if h in prog_candidates), 0)
        hour_idx = header_lc.index("hour")
        rating_idx = header_lc.index("rating")

        # initialize empty arrays
        for row in reader:
            if not row or all([c.strip()=="" for c in row]):
                continue
            try:
                prog = row[prog_idx].strip()
                hour_raw = row[hour_idx].strip()
                # try parse hour like '6' or '06' or '06:00'
                hour = None
                if ":" in hour_raw:
                    hour = int(hour_raw.split(":")[0])
                else:
                    hour = int(hour_raw)
                rating_val = float(row[rating_idx]) if row[rating_idx].strip() != "" else 0.0
            except Exception:
                # skip malformed line
                continue

            if prog not in results:
                results[prog] = [0.0]*18
            if 6 <= hour <= 23:
                idx = hour - 6
                results[prog][idx] = rating_val
        return results

    # else assume wide format: first col program, next columns ratings (sequence)
    # parse each row: first column program, remaining columns numeric ratings
    for row in reader:
        if not row:
            continue
        prog = row[0].strip()
        if prog == "":
            continue
        ratings = []
        for val in row[1:]:
            try:
                ratings.append(float(val) if val != "" else 0.0)
            except:
                ratings.append(0.0)
        # pad/truncate to 18
        if len(ratings) >= 18:
            ratings = ratings[:18]
        else:
            ratings = ratings + [0.0] * (18 - len(ratings))
        results[prog] = ratings

    return results

def read_csv_from_fileobj(fileobj):
    text = fileobj.getvalue().decode("utf-8")
    return parse_csv_text(text)

def read_csv_from_path(path):
    try:
        with open(path, mode="r", encoding="utf-8", newline="") as f:
            text = f.read()
    except FileNotFoundError:
        return {}
    return parse_csv_text(text)

# -------------------- Normalize ratings to exactly 18 columns (06:00-23:00) --------------------
def normalize_ratings(ratings_dict, target_len=18):
    normalized = {}
    for prog, arr in ratings_dict.items():
        if len(arr) >= target_len:
            normalized[prog] = arr[:target_len]
        else:
            padded = arr + [0.0] * (target_len - len(arr))
            normalized[prog] = padded
    return normalized

# -------------------- Genetic Algorithm Utilities --------------------
def fitness_function(schedule, ratings):
    total_rating = 0.0
    for t, program in enumerate(schedule):
        total_rating += ratings.get(program, [0.0]*18)[t]
    return total_rating

def initialize_population(programs, pop_size, slot_count):
    population = []
    for _ in range(pop_size):
        schedule = [random.choice(programs) for _ in range(slot_count)]
        population.append(schedule)
    return population

def crossover(schedule1, schedule2):
    if len(schedule1) < 3:
        return schedule1.copy(), schedule2.copy()
    point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2

def mutate(schedule, all_programs):
    idx = random.randint(0, len(schedule) - 1)
    schedule[idx] = random.choice(all_programs)
    return schedule

def genetic_algorithm(ratings, all_programs, slot_count, generations, pop_size, crossover_rate, mutation_rate, elitism):
    population = initialize_population(all_programs, pop_size, slot_count)

    for _ in range(generations):
        population.sort(key=lambda s: fitness_function(s, ratings), reverse=True)
        new_pop = population[:elitism]

        while len(new_pop) < pop_size:
            parent1, parent2 = random.choices(population, k=2)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                mutate(child1, all_programs)
            if random.random() < mutation_rate:
                mutate(child2, all_programs)

            new_pop.extend([child1, child2])

        population = new_pop[:pop_size]

    best = max(population, key=lambda s: fitness_function(s, ratings))
    return best

# -------------------- Inputs: Uploader + Defaults --------------------
st.write("Upload CSV (optional). Supported formats: wide (Program, r06..r23) OR long (program,hour,rating).")
uploaded = st.file_uploader("Upload program_ratings CSV", type=["csv"])

if uploaded:
    raw_ratings = read_csv_from_fileobj(uploaded)
else:
    raw_ratings = read_csv_from_path("program_ratings_modified.csv")

if not raw_ratings:
    st.error("No ratings loaded. Upload a CSV or place 'program_ratings_modified.csv' beside this app.")
    st.stop()

# ensure exactly 18 rating columns per program
ratings = normalize_ratings(raw_ratings, target_len=18)

# -------------------- Fixed time slots --------------------
time_slots = list(range(6, 24))  # 06:00 ... 23:00 (18 slots)
slot_count = len(time_slots)

# -------------------- Sidebar: GA parameters (3 trials) --------------------
st.sidebar.header("GA Parameters (3 Trials)")
st.sidebar.write("Crossover range: 0.00 – 0.95 | Mutation range: 0.01 – 0.05")

# Trial 1
st.sidebar.subheader("Trial 1")
co1 = st.sidebar.slider("Crossover (T1)", min_value=0.0, max_value=0.95, value=0.80, step=0.05, key="co1")
mu1 = st.sidebar.slider("Mutation (T1)", min_value=0.01, max_value=0.05, value=0.02, step=0.01, key="mu1")

# Trial 2
st.sidebar.subheader("Trial 2")
co2 = st.sidebar.slider("Crossover (T2)", min_value=0.0, max_value=0.95, value=0.80, step=0.05, key="co2")
mu2 = st.sidebar.slider("Mutation (T2)", min_value=0.01, max_value=0.05, value=0.02, step=0.01, key="mu2")

# Trial 3
st.sidebar.subheader("Trial 3")
co3 = st.sidebar.slider("Crossover (T3)", min_value=0.0, max_value=0.95, value=0.80, step=0.05, key="co3")
mu3 = st.sidebar.slider("Mutation (T3)", min_value=0.01, max_value=0.05, value=0.02, step=0.01, key="mu3")

# Other GA controls
st.sidebar.subheader("Other")
generations = st.sidebar.number_input("Generations", min_value=10, max_value=2000, value=100, step=10)
population = st.sidebar.number_input("Population size", min_value=10, max_value=1000, value=100, step=10)
elitism = st.sidebar.number_input("Elitism (top-k)", min_value=1, max_value=10, value=2, step=1)

# Show sample of loaded data (first 5 programs)
st.write("### Sample programs (first 5 rows)")
sample_preview = []
for i, (prog, arr) in enumerate(ratings.items()):
    if i >= 5:
        break
    sample_preview.append({"Program": prog, "Ratings (06..23)": ", ".join([f"{x:.1f}" for x in arr[:6]]) + " ..."})
sample_df = pd.DataFrame(sample_preview)
st.dataframe(sample_df)

# -------------------- Run GA button --------------------
if st.button("Run All 3 Trials"):
    trials = [
        ("Trial 1", co1, mu1),
        ("Trial 2", co2, mu2),
        ("Trial 3", co3, mu3),
    ]

    all_programs = list(ratings.keys())
    if not all_programs:
        st.error("No programs available to schedule.")
        st.stop()

    for name, co_rate, mut_rate in trials:
        best_schedule = genetic_algorithm(
            ratings=ratings,
            all_programs=all_programs,
            slot_count=slot_count,
            generations=int(generations),
            pop_size=int(population),
            crossover_rate=float(co_rate),
            mutation_rate=float(mut_rate),
            elitism=int(elitism),
        )

        total = fitness_function(best_schedule, ratings)

        st.subheader(name)
        st.write(f"Parameters used — Crossover: {co_rate} | Mutation: {mut_rate}")
        # show total fitness with 1 decimal
        st.write(f"Total Fitness: {total:.1f}")

        # Build display table guaranteed 18 rows (06:00-23:00)
        rows = []
        for idx in range(slot_count):
            program = best_schedule[idx]
            rating = ratings.get(program, [0.0]*slot_count)[idx]
            # FORCE 1 decimal string so Streamlit won't re-format it
            rows.append({
                "Time Slot": f"{time_slots[idx]:02d}:00",
                "Program": program,
                "Rating": format(rating, ".1f")
            })

        df_out = pd.DataFrame(rows)
        st.table(df_out)   # static table display
        st.markdown("---")
