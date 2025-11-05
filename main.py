import streamlit as st
import csv
import random
import pandas as pd

# ===================== STEP 1: READ CSV (AUTO LOAD) =====================

def read_csv_to_dict(file_path):
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)  # skip header
            for row in reader:
                program = row[0]
                ratings = [float(x) if x else 0.0 for x in row[1:]]
                program_ratings[program] = ratings
    except FileNotFoundError:
        st.error("❌ File 'program_ratings_modified.csv' not found. Please make sure it’s in the same folder as this app.")
    return program_ratings


# ===================== STEP 2: GENETIC ALGORITHM FUNCTIONS =====================

def fitness_function(schedule, ratings):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings and time_slot < len(ratings[program]):
            total_rating += ratings[program][time_slot]
    return total_rating


def initialize_population(programs, pop_size):
    population = []
    for _ in range(pop_size):
        schedule = programs.copy()
        random.shuffle(schedule)
        population.append(schedule)
    return population


def crossover(schedule1, schedule2):
    point = random.randint(1, len(schedule1) - 2)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2


def mutate(schedule, all_programs):
    idx = random.randint(0, len(schedule) - 1)
    new_program = random.choice(all_programs)
    schedule[idx] = new_program
    return schedule


def genetic_algorithm(ratings, all_programs, generations, pop_size, crossover_rate, mutation_rate, elitism):
    population = initialize_population(all_programs, pop_size)
    best_fitness_history = []

    for generation in range(generations):
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
        best_fitness_history.append(fitness_function(population[0], ratings))

    best = max(population, key=lambda s: fitness_function(s, ratings))
    return best, best_fitness_history


# ===================== STREAMLIT INTERFACE =====================

st.title("📺 Genetic Algorithm TV Schedule Optimizer")

st.sidebar.header("⚙️ Genetic Algorithm Parameters")

file_path = "program_ratings_modified.csv"
ratings = read_csv_to_dict(file_path)

if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 6 + len(all_programs)))

    # ===================== TRIAL 1 PARAMETERS =====================
    st.sidebar.subheader("Trial 1 Parameters")
    CO_R1 = st.sidebar.slider("Crossover Rate (Trial 1)", 0.0, 1.0, 0.8, step=0.05, key="co1")
    MUT_R1 = st.sidebar.slider("Mutation Rate (Trial 1)", 0.0, 1.0, 0.2, step=0.01, key="mu1")

    # ===================== TRIAL 2 PARAMETERS =====================
    st.sidebar.subheader("Trial 2 Parameters")
    CO_R2 = st.sidebar.slider("Crossover Rate (Trial 2)", 0.0, 1.0, 0.8, step=0.05, key="co2")
    MUT_R2 = st.sidebar.slider("Mutation Rate (Trial 2)", 0.0, 1.0, 0.2, step=0.01, key="mu2")

    # ===================== TRIAL 3 PARAMETERS =====================
    st.sidebar.subheader("Trial 3 Parameters")
    CO_R3 = st.sidebar.slider("Crossover Rate (Trial 3)", 0.0, 1.0, 0.8, step=0.05, key="co3")
    MUT_R3 = st.sidebar.slider("Mutation Rate (Trial 3)", 0.0, 1.0, 0.2, step=0.01, key="mu3")

    # Fixed GA parameters
    GEN = 100
    POP = 100
    EL_S = 2

    st.write("### Loaded Programs (Sample)")
    sample_df = pd.DataFrame(list(ratings.items()), columns=["Program", "Ratings"]).head(5)
    st.dataframe(sample_df)

    if st.button("🚀 Run All 3 Trials"):
        trial_settings = [
            ("Trial 1", CO_R1, MUT_R1),
            ("Trial 2", CO_R2, MUT_R2),
            ("Trial 3", CO_R3, MUT_R3),
        ]

        results = []

        with st.spinner("Running all 3 trials..."):
            for name, co_rate, mut_rate in trial_settings:
                best_schedule, fitness_history = genetic_algorithm(
                    ratings, all_programs, GEN, POP, co_rate, mut_rate, EL_S
                )
                total_fitness = fitness_function(best_schedule, ratings)
                results.append((name, co_rate, mut_rate, best_schedule, fitness_history, total_fitness))

        # Display all trials
        for name, co_rate, mut_rate, schedule, fitness_history, total_fit in results:
            st.markdown(f"## 🧪 {name}")
            st.write(f"**Crossover Rate:** {co_rate} | **Mutation Rate:** {mut_rate}")
            st.metric(label="Total Fitness", value=round(total_fit, 2))

            # Display schedule
            schedule_data = []
            for i, program in enumerate(schedule):
                schedule_data.append({
                    "Time Slot": f"{all_time_slots[i]:02d}:00",
                    "Program": program,
                    "Rating": round(ratings[program][i], 2) if i < len(ratings[program]) else "-"
                })
            st.dataframe(pd.DataFrame(schedule_data))

            # Display fitness chart
            st.line_chart(fitness_history)
            st.divider()
