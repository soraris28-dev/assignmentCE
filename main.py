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

# -------------------- Helper: Read CSV --------------------
def read_csv_from_fileobj(fileobj):
    text = fileobj.getvalue().decode("utf-8")
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    program_ratings = {}
    for row in reader:
        if not row:
            continue
        program = row[0]
        ratings = []
        for val in row[1:]:
            try:
                ratings.append(float(val) if val != "" else 0.0)
            except:
                ratings.append(0.0)
        program_ratings[program] = ratings
    return program_ratings

def read_csv_from_path(path):
    program_ratings = {}
    try:
        with open(path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                if not row:
                    continue
                program = row[0]
                ratings = []
                for val in row[1:]:
                    try:
                        ratings.append(float(val) if val != "" else 0.0)
                    except:
                        ratings.append(0.0)
                program_ratings[program] = ratings
    except FileNotFoundError:
        return {}
    return program_ratings

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
st.write("Upload CSV (optional). Format: Program, r06, r07, ..., r23  (18 rating columns).")
uploaded = st.file_uploader("Upload program_ratings CSV", type=["csv"])

if uploaded:
    raw_ratings = read_csv_from_fileobj(uploaded)
else:
    raw_ratings = read_csv_from_path("program_ratings_modified.csv")

if not raw_ratings:
    st.error("No ratings loaded. Upload a CSV or place 'program_ratings_modified.csv' beside this app.")
    st.stop()

ratings = normalize_ratings(raw_ratings, target_len=18)

# -------------------- Fixed time slots --------------------
time_slots = list(range(6, 24))
slot_count = len(time_slots)

# -------------------- Sidebar: GA parameters (3 Trials) --------------------
st.sidebar.header("GA Parameters (3 Trials)")
st.sidebar.write("Crossover range: 0.00 – 0.95 | Mutation range: 0.01 – 0.05")

st.sidebar.subheader("Trial 1")
co1 = st.sidebar.slider("Crossover (T1)", 0.0, 0.95, 0.80, 0.05, key="co1")
mu1 = st.sidebar.slider("Mutation (T1)", 0.01, 0.05, 0.02, 0.01, key="mu1")

st.sidebar.subheader("Trial 2")
co2 = st.sidebar.slider("Crossover (T2)", 0.0, 0.95, 0.80, 0.05, key="co2")
mu2 = st.sidebar.slider("Mutation (T2)", 0.01, 0.05, 0.02, 0.01, key="mu2")

st.sidebar.subheader("Trial 3")
co3 = st.sidebar.slider("Crossover (T3)", 0.0, 0.95, 0.80, 0.05, key="co3")
mu3 = st.sidebar.slider("Mutation (T3)", 0.01, 0.05, 0.02, 0.01, key="mu3")

st.sidebar.subheader("Other")
generations = st.sidebar.number_input("Generations", 10, 2000, 100, 10)
population = st.sidebar.number_input("Population size", 10, 1000, 100, 10)
elitism = st.sidebar.number_input("Elitism (top-k)", 1, 10, 2, 1)

# Show sample of loaded data
st.write("### Sample programs (first 5 rows)")
sample_df = pd.DataFrame(list(ratings.items()), columns=["Program", "Ratings"]).head(5)
st.dataframe(sample_df)

# -------------------- Run GA button --------------------
if st.button("Run All 3 Trials"):
    trials = [
        ("Trial 1", co1, mu1),
        ("Trial 2", co2, mu2),
        ("Trial 3", co3, mu3),
    ]

    all_programs = list(ratings.keys())

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
        st.write(f"Total Fitness: {total:.3f}")

        rows = []
        for idx in range(slot_count):
            program = best_schedule[idx]
            rating = ratings.get(program, [0.0]*slot_count)[idx]
            rows.append({
                "Time Slot": f"{time_slots[idx]:02d}:00",
                "Program": program,
                "Rating": format(rating, ".1f")  #  << FORCE 1 decimal only
            })

        df_out = pd.DataFrame(rows)
        st.table(df_out)
        st.markdown("---")
