# --- IMPORTS ---
import pandas as pd
import random
import time
import math
import copy
import itertools
import openpyxl
import gurobipy as gp
from gurobipy import GRB

# --- UTILITIES ---
def generate_random_input_table(n, q, min_time=10, max_time=50):
    return [[random.randint(min_time, max_time) for _ in range(n)] for _ in range(q)]

def calculate_objective_value(input_table, pay_table):
    num_jobs = len(input_table[0])
    cumulative = [0] * num_jobs
    for d, day in enumerate(pay_table):
        current_time = 0
        for job in day:
            p = input_table[d][job - 1]
            current_time += p
            cumulative[job - 1] += current_time
    return max(cumulative)

def calculate_spt_lower_bound(input_table):
    q = len(input_table)
    n = len(input_table[0])
    total = 0
    for day in input_table:
        spt = sorted(day)
        total += sum((n - j - 1) * spt[j] for j in range(n))
    return math.ceil(total / n)

def pay_table_to_string(pay_table):
    return ' | '.join(['[' + ','.join(map(str, day)) + ']' for day in pay_table])

# --- LP_GUROBI CLASS ---
class LP_Gurobi:
    def __init__(self, time_limit_seconds):
        self.time_limit_seconds = time_limit_seconds
        self.optimal_solution = None
        self.objective_value = None
        self.best_bound = None
        self.optimization_runtime = None
        self.gap = None
        self.pay_table = []

    def calc_pay_and_value(self, table):
        num_rows = len(table)
        num_cols = len(table[0])

        model = gp.Model("LP_Fair_Scheduling")
        model.Params.OutputFlag = 1
        model.Params.TimeLimit = self.time_limit_seconds

        x = {}
        for k in range(1, num_rows + 1):
            for i in range(0, num_cols + 1):
                for j in range(1, num_cols + 1):
                    if i != j:
                        x[i, j, k] = model.addVar(vtype=GRB.BINARY, name=f"X{i}{j}{k}")

        c = {}
        for j in range(1, num_cols + 1):
            for k in range(1, num_rows + 1):
                c[j, k] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"C{j}{k}")

        Z = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="Z")

        M = max(sum(row) for row in table)

        model.setObjective(Z, GRB.MINIMIZE)

        for k in range(1, num_rows + 1):
            for i in range(1, num_cols + 1):
                for j in range(1, num_cols + 1):
                    if i != j:
                        model.addConstr(c[i, k] + table[k-1][j-1] - M * (1 - x[i, j, k]) <= c[j, k])
                        model.addConstr(c[j, k] <= c[i, k] + table[k-1][j-1] + M * (1 - x[i, j, k]))

        for k in range(1, num_rows + 1):
            for j in range(1, num_cols + 1):
                model.addConstr(table[k-1][j-1] - M * (1 - x[0, j, k]) <= c[j, k])
                model.addConstr(c[j, k] <= table[k-1][j-1] + M * (1 - x[0, j, k]))

        for k in range(1, num_rows + 1):
            for j in range(1, num_cols + 1):
                model.addConstr(gp.quicksum(x[i, j, k] for i in range(0, num_cols + 1) if (i, j, k) in x) == 1)

        for k in range(1, num_rows + 1):
            model.addConstr(gp.quicksum(x[i, j, k] for i in range(1, num_cols + 1) for j in range(1, num_cols + 1) if (i, j, k) in x) == num_cols - 1)

        for k in range(1, num_rows + 1):
            for i in range(1, num_cols + 1):
                model.addConstr(gp.quicksum(x[i, j, k] for j in range(1, num_cols + 1) if (i, j, k) in x) <= 1)

        for i in range(1, num_cols + 1):
            model.addConstr(gp.quicksum(c[i, k] for k in range(1, num_rows + 1)) <= Z)

        model.optimize()

        if model.SolCount == 0:
            self.objective_value = float('inf')
            self.best_bound = 0
            self.optimization_runtime = model.Runtime
            self.optimal_solution = False
            self.gap = 1
            self.pay_table = [['NO_SOLUTION']] * num_rows
            return

        self.optimal_solution = model.Status == GRB.OPTIMAL
        self.objective_value = model.ObjVal
        self.best_bound = model.ObjBound
        self.optimization_runtime = model.Runtime
        self.gap = 0 if self.optimal_solution else (self.objective_value - self.best_bound) / self.objective_value

        self.extract_pay_table(x, table)

    def extract_pay_table(self, x, table):
        num_rows = len(table)
        num_cols = len(table[0])
        grouped_by_day = {}

        for (i, j, k), var in x.items():
            if var.X > 0.5:
                if k not in grouped_by_day:
                    grouped_by_day[k] = []
                grouped_by_day[k].append((i, j))

        self.pay_table = []
        for k in sorted(grouped_by_day.keys()):
            current_job = 0
            sequence = []
            while True:
                next_job = next((j for i, j in grouped_by_day[k] if i == current_job), None)
                if next_job is None:
                    break
                sequence.append(next_job)
                current_job = next_job
            self.pay_table.append(sequence)

    def get_pay_table(self):
        return self.pay_table

    def get_optimal_solution(self):
        return self.optimal_solution

    def get_sol_value(self):
        return self.objective_value

    def get_best_bound(self):
        return self.best_bound

    def get_runtime(self):
        return self.optimization_runtime

    def get_gap(self):
        return self.gap


class H1_alg:
    def __init__(self):
        self.pay_table = None
        self.solution_value = None
        self.runtime = None

    def _optimize_first_two_days(self, two_days_df):
        S1, S2 = [], []
        for index, row in two_days_df.iterrows():
            if row[1] >= row[0]:
                S1.append(index + 1)
            else:
                S2.append(index + 1)

        S1 = sorted(S1, key=lambda job: two_days_df.iloc[job - 1, 0])
        S2 = sorted(S2, key=lambda job: two_days_df.iloc[job - 1, 1], reverse=True)

        pay_day1 = S1 + S2
        pay_day2 = list(reversed(pay_day1))
        return pay_day1, pay_day2

    def _simulate_day(self, schedule, day_data, total_completion):
        completion_times = []
        cumulative_time = 0
        for job in schedule:
            processing_time = day_data[job - 1]
            cumulative_time += processing_time
            total_completion[job] += cumulative_time
            completion_times.append((job, processing_time, cumulative_time, total_completion[job]))
        return completion_times

    def calc_pay_and_value(self, input_table):
        start_time = time.time()
        df = pd.DataFrame(input_table)
        num_days = df.shape[0]
        num_jobs = df.shape[1]

        self.pay_table = []
        total_completion = {job: 0 for job in range(1, num_jobs + 1)}

        # Days 1 & 2
        first_two_days = df.iloc[:2].transpose()
        pay_day1, pay_day2 = self._optimize_first_two_days(first_two_days)
        self.pay_table.append(pay_day1)
        self.pay_table.append(pay_day2)

        # Simulate days 1 & 2
        self._simulate_day(pay_day1, df.iloc[0], total_completion)
        self._simulate_day(pay_day2, df.iloc[1], total_completion)

        # Remaining days: greedy by total completion
        for day_index in range(2, num_days):
            next_schedule = sorted(total_completion, key=total_completion.get, reverse=True)
            self.pay_table.append(next_schedule)
            self._simulate_day(next_schedule, df.iloc[day_index], total_completion)

        self.solution_value = max(total_completion.values())
        self.runtime = time.time() - start_time

    def get_pay_table(self, input_table):
        if self.pay_table is None:
            self.calc_pay_and_value(input_table)
        return self.pay_table

    def get_sol_value(self, input_table):
        if self.solution_value is None:
            self.calc_pay_and_value(input_table)
        return self.solution_value

    def get_runtime(self, input_table):
        if self.runtime is None:
            self.calc_pay_and_value(input_table)
        return self.runtime
    
class H2_alg:
    def __init__(self):
        self.pay_table = None
        self.solution_value = None
        self.runtime = None

    def _optimize_two_days(self, two_days_df):
        S1, S2 = [], []
        for index, row in two_days_df.iterrows():
            if row.iloc[1] >= row.iloc[0]:  # ✅ FIXED: safe position-based indexing
                S1.append(index + 1)
            else:
                S2.append(index + 1)
        S1 = sorted(S1, key=lambda job: two_days_df.iloc[job - 1, 0])
        S2 = sorted(S2, key=lambda job: two_days_df.iloc[job - 1, 1], reverse=True)
        return S1 + S2, list(reversed(S1 + S2))

    def _simulate_day(self, schedule, day_data, total_completion):
        completion_times = []
        cumulative_time = 0
        for job in schedule:
            processing_time = day_data[job - 1]
            cumulative_time += processing_time
            total_completion[job] += cumulative_time
            completion_times.append((job, processing_time, cumulative_time, total_completion[job]))
        return completion_times

    def calc_pay_and_value(self, input_table):
        start_time = time.time()
        df = pd.DataFrame(input_table)
        num_days = df.shape[0]
        num_jobs = df.shape[1]

        self.pay_table = []
        total_completion = {job: 0 for job in range(1, num_jobs + 1)}

        # Handle pairs of days
        for day_index in range(0, num_days - 1, 2):
            two_days_df = df.iloc[day_index:day_index + 2].transpose()
            pay1, pay2 = self._optimize_two_days(two_days_df)
            self.pay_table.append(pay1)
            self.pay_table.append(pay2)
            self._simulate_day(pay1, df.iloc[day_index], total_completion)
            self._simulate_day(pay2, df.iloc[day_index + 1], total_completion)

        # Handle last day if number of days is odd
        if num_days % 2 == 1:
            last_day = num_days - 1
            greedy_order = sorted(total_completion, key=total_completion.get, reverse=True)
            self.pay_table.append(greedy_order)
            self._simulate_day(greedy_order, df.iloc[last_day], total_completion)

        self.solution_value = max(total_completion.values())
        self.runtime = time.time() - start_time

    def get_pay_table(self, input_table):
        if self.pay_table is None:
            self.calc_pay_and_value(input_table)
        return self.pay_table

    def get_sol_value(self, input_table):
        if self.solution_value is None:
            self.calc_pay_and_value(input_table)
        return self.solution_value

    def get_runtime(self, input_table):
        if self.runtime is None:
            self.calc_pay_and_value(input_table)
        return self.runtime
    
    

class Meta_insertion_algH1:
    def __init__(self):
        self.pay_table = None
        self.solution_value = None
        self.runtime = None

    def _optimize_first_two_days(self, two_days_df):
        S1, S2 = [], []
        for index, row in two_days_df.iterrows():
            if row[1] >= row[0]:
                S1.append(index + 1)
            else:
                S2.append(index + 1)
        S1 = sorted(S1, key=lambda job: two_days_df.iloc[job - 1, 0])
        S2 = sorted(S2, key=lambda job: two_days_df.iloc[job - 1, 1], reverse=True)
        return S1 + S2, list(reversed(S1 + S2))

    def _simulate_day(self, schedule, day_data, total_completion):
        cumulative_time = 0
        results = []
        for job in schedule:
            process_time = day_data[job - 1]
            cumulative_time += process_time
            total_completion[job] += cumulative_time
            results.append((job, process_time, cumulative_time, total_completion[job]))
        return results

    def _evaluate_schedule(self, input_df):
        pay_table = []
        total_completion = {j + 1: 0 for j in range(input_df.shape[1])}
        pay1, pay2 = self._optimize_first_two_days(input_df.iloc[:2].transpose())

        pay_table.append(pay1)
        pay_table.append(pay2)

        self._simulate_day(pay1, input_df.iloc[0], total_completion)
        self._simulate_day(pay2, input_df.iloc[1], total_completion)

        for day_index in range(2, input_df.shape[0]):
            day_order = sorted(total_completion, key=total_completion.get, reverse=True)
            pay_table.append(day_order)
            self._simulate_day(day_order, input_df.iloc[day_index], total_completion)

        return pay_table, max(total_completion.values())

    def calc_pay_and_value(self, input_table):
        start_time = time.time()
        df = pd.DataFrame(input_table)
        num_days = df.shape[0]

        best_order = [0, 1]
        _, best_val = self._evaluate_schedule(df.iloc[best_order].reset_index(drop=True))

        for new_day in range(2, num_days):
            best_pos = None
            best_val_for_day = float('inf')

            for pos in range(len(best_order) + 1):
                candidate_order = best_order[:pos] + [new_day] + best_order[pos:]
                reordered_df = df.iloc[candidate_order].reset_index(drop=True)
                _, val = self._evaluate_schedule(reordered_df)

                if val < best_val_for_day:
                    best_val_for_day = val
                    best_pos = candidate_order

            best_order = best_pos

        # Final schedule evaluation
        final_df = df.iloc[best_order].reset_index(drop=True)
        final_pay_table, final_val = self._evaluate_schedule(final_df)

        # Reorder pay_table back to original day positions
        reordered = [None] * len(best_order)
        for i, orig_day in enumerate(best_order):
            reordered[orig_day] = final_pay_table[i]

        self.pay_table = reordered
        self.solution_value = final_val
        self.runtime = time.time() - start_time

    def get_pay_table(self, input_table):
        if self.pay_table is None:
            self.calc_pay_and_value(input_table)
        return self.pay_table

    def get_sol_value(self, input_table):
        if self.solution_value is None:
            self.calc_pay_and_value(input_table)
        return self.solution_value

    def get_runtime(self, input_table):
        if self.runtime is None:
            self.calc_pay_and_value(input_table)
        return self.runtime
    
    
class Meta_insertion_algH2:
    def __init__(self):
        self.pay_table = None
        self.solution_value = None
        self.runtime = None

    def _optimize_two_days(self, two_days_df):
        S1, S2 = [], []
        for index, row in two_days_df.iterrows():
            if row.iloc[1] >= row.iloc[0]:
                S1.append(index + 1)
            else:
                S2.append(index + 1)
        S1 = sorted(S1, key=lambda job: two_days_df.iloc[job - 1, 0])
        S2 = sorted(S2, key=lambda job: two_days_df.iloc[job - 1, 1], reverse=True)
        return S1 + S2, list(reversed(S1 + S2))

    def _simulate_day(self, schedule, day_data, total_completion):
        cumulative_time = 0
        results = []
        for job in schedule:
            processing_time = day_data[job - 1]
            cumulative_time += processing_time
            total_completion[job] += cumulative_time
            results.append((job, processing_time, cumulative_time, total_completion[job]))
        return results

    def _evaluate_schedule(self, input_df):
        total_completion = {j + 1: 0 for j in range(input_df.shape[1])}
        gantt_chart = []
        num_days = input_df.shape[0]

        for day_index in range(0, num_days - 1, 2):
            two_days_df = input_df.iloc[day_index:day_index + 2].transpose()
            pay1, pay2 = self._optimize_two_days(two_days_df)
            gantt_chart.append((f"day{day_index + 1}", self._simulate_day(pay1, input_df.iloc[day_index], total_completion)))
            gantt_chart.append((f"day{day_index + 2}", self._simulate_day(pay2, input_df.iloc[day_index + 1], total_completion)))

        if num_days % 2 == 1:
            last_day = num_days - 1
            greedy_order = sorted(total_completion, key=total_completion.get, reverse=True)
            gantt_chart.append((f"day{last_day + 1}", self._simulate_day(greedy_order, input_df.iloc[last_day], total_completion)))

        return gantt_chart, max(total_completion.values())

    def calc_pay_and_value(self, input_table):
        start_time = time.time()
        df = pd.DataFrame(input_table)
        num_days = df.shape[0]

        best_order = [0, 1]
        _, best_val = self._evaluate_schedule(df.iloc[best_order].reset_index(drop=True))

        for new_day in range(2, num_days):
            best_val_for_day = float('inf')
            best_order_for_day = None

            for pos in range(len(best_order) + 1):
                candidate_order = best_order[:pos] + [new_day] + best_order[pos:]
                reordered_df = df.iloc[candidate_order].reset_index(drop=True)
                _, val = self._evaluate_schedule(reordered_df)

                if val < best_val_for_day:
                    best_val_for_day = val
                    best_order_for_day = candidate_order

            best_order = best_order_for_day

        # Final schedule evaluation
        final_df = df.iloc[best_order].reset_index(drop=True)
        final_gantt_chart, final_val = self._evaluate_schedule(final_df)

        # Reorder pay table back to original day positions
        pay_temp_table = [[job[0] for job in day[1]] for day in final_gantt_chart]
        reordered = [None] * len(best_order)
        for i, orig_day in enumerate(best_order):
            reordered[orig_day] = pay_temp_table[i]

        self.pay_table = reordered
        self.solution_value = final_val
        self.runtime = time.time() - start_time

    def get_pay_table(self, input_table):
        if self.pay_table is None:
            self.calc_pay_and_value(input_table)
        return self.pay_table

    def get_sol_value(self, input_table):
        if self.solution_value is None:
            self.calc_pay_and_value(input_table)
        return self.solution_value

    def get_runtime(self, input_table):
        if self.runtime is None:
            self.calc_pay_and_value(input_table)
        return self.runtime
    
class SA_alg:
    def __init__(self):
        self.initial_temperature = 10000
        self.cooling_rate = 0.95
        self.iterations = 10

        self.pay_table = None
        self.solution_value = None
        self.runtime = None

    def _create_neighbor(self, pay_table, jobs_with_max):
        job_to_move = random.choice(jobs_with_max)
        eligible_days = [d for d, row in enumerate(pay_table) if row[0] != job_to_move]
        if not eligible_days:
            return pay_table  # Can't swap

        day = random.choice(eligible_days)
        day_row = pay_table[day]
        i = day_row.index(job_to_move)
        if i == 0:
            return pay_table  # Already first or second

        swap_candidate = random.choice(day_row[:i])
        j = day_row.index(swap_candidate)
        day_row[i], day_row[j] = day_row[j], day_row[i]
        return pay_table

    def _evaluate_schedule(self, input_table, pay_table):
        num_jobs = len(input_table[0])
        cumulative = [0] * num_jobs

        for d, day in enumerate(pay_table):
            current_time = 0
            for job in day:
                proc = input_table[d][job - 1]
                current_time += proc
                cumulative[job - 1] += current_time

        max_c = max(cumulative)
        critical_jobs = [j + 1 for j, val in enumerate(cumulative) if val == max_c]
        return max_c, critical_jobs

    def calc_pay_and_value(self, input_table, initial_pay_table):
        start_time = time.time()
        cur_pay_table = copy.deepcopy(initial_pay_table)
        best_pay_table = copy.deepcopy(initial_pay_table)

        cur_sol, critical_jobs = self._evaluate_schedule(input_table, cur_pay_table)
        best_sol = cur_sol

        temperature = self.initial_temperature

        while temperature > 1:
            for _ in range(self.iterations):
                tested_pay_table = copy.deepcopy(self._create_neighbor(copy.deepcopy(cur_pay_table), critical_jobs))
                tested_sol, _ = self._evaluate_schedule(input_table, tested_pay_table)

                delta = tested_sol - cur_sol
                accept_prob = math.exp(-delta / temperature) if delta > 0 else 1

                if delta < 0 or random.random() < accept_prob:
                    cur_pay_table = copy.deepcopy(tested_pay_table)
                    cur_sol = tested_sol
                    _, critical_jobs = self._evaluate_schedule(input_table, cur_pay_table)

                    if cur_sol < best_sol:
                        best_sol = cur_sol
                        best_pay_table = copy.deepcopy(cur_pay_table)

            temperature *= self.cooling_rate

        self.pay_table = best_pay_table
        self.solution_value = best_sol
        self.runtime = time.time() - start_time

    def get_pay_table(self, input_table, pay_table):
        if self.pay_table is None:
            self.calc_pay_and_value(input_table, pay_table)
        return self.pay_table

    def get_sol_value(self, input_table, pay_table):
        if self.solution_value is None:
            self.calc_pay_and_value(input_table, pay_table)
        return self.solution_value

    def get_runtime(self, input_table, pay_table):
        if self.runtime is None:
            self.calc_pay_and_value(input_table, pay_table)
        return self.runtime
    
# --- EXPERIMENT RUNNER ---
def run_experiment(n_values, q_values, instances_per_setting, time_limit_minutes):
    all_results = []
    for n, q in itertools.product(n_values, q_values):
        for instance_id in range(1, instances_per_setting + 1):
            print(f"\n🧪 Running instance {instance_id} for (n={n}, q={q})")
            input_table = generate_random_input_table(n, q)
            print("Input table:")
            for row in input_table:
                print(row)

            # Run 4 heuristics
            algos = [H1_alg(), H2_alg(), Meta_insertion_algH1(), Meta_insertion_algH2()]
            results = []
            for alg in algos:
                sol = alg.get_sol_value(input_table)
                pay = alg.get_pay_table(input_table)
                runtime = alg.get_runtime(input_table)
                results.append({'name': alg.__class__.__name__, 'sol': sol, 'pay': pay, 'runtime': runtime})

            # Best heuristic result
            best_result = min(results, key=lambda x: x['sol'])

            # Run SA
            sa = SA_alg()
            sa_sol = sa.get_sol_value(input_table, best_result['pay'])
            sa_pay = sa.get_pay_table(input_table, best_result['pay'])
            sa_runtime = sa.get_runtime(input_table, best_result['pay'])
            results.append({'name': 'SA_alg', 'sol': sa_sol, 'pay': sa_pay, 'runtime': sa_runtime})

            # Run LP_Gurobi with time limit in seconds
            lp = LP_Gurobi(time_limit_seconds=time_limit_minutes * 60)
            lp.calc_pay_and_value(input_table)
            lp_sol = lp.get_sol_value()
            lp_pay = lp.get_pay_table()
            lp_runtime = lp.get_runtime()
            lp_bound = lp.get_best_bound()
            results.append({
                'name': 'LP_Gurobi',
                'sol': lp_sol,
                'pay': lp_pay,
                'runtime': lp_runtime,
                'bound': lp_bound
            })

            spt_lb = calculate_spt_lower_bound(input_table)

            for res in results:
                gap_spt = (res['sol'] - spt_lb) / res['sol'] if res['sol'] > 0 else 0
                gap_lp = (res['sol'] - lp_bound) / res['sol'] if res['sol'] > 0 else 0
                all_results.append({
                    'n': n,
                    'q': q,
                    'instance': instance_id,
                    'algorithm': res['name'],
                    'objective_value': res['sol'],
                    'runtime': res['runtime'],
                    'gurobi_bound': lp_bound,
                    'gap_lp': gap_lp,
                    'spt_bound': spt_lb,
                    'gap_spt': gap_spt,
                    'pay_table': pay_table_to_string(res['pay']),
                    'input_table': str(input_table)
                })

    return pd.DataFrame(all_results)


# --- EXECUTE CONFIGURATION ---
n_values = [75]
q_values = [30, 50]
instances_per_setting = 2
time_limit_minutes = 120

# --- RUN & EXPORT TO EXCEL ---
df = run_experiment(n_values, q_values, instances_per_setting, time_limit_minutes)
df.to_excel("experiment_results_full.xlsx", index=False)

agg = df.groupby(['n', 'q', 'algorithm']).agg(
    avg_gap_spt=('gap_spt', 'mean'),
    max_gap_spt=('gap_spt', 'max'),
    avg_gap_lp=('gap_lp', 'mean'),
    max_gap_lp=('gap_lp', 'max'),
    avg_runtime=('runtime', 'mean')
).reset_index()

agg.to_excel("experiment_summary.xlsx", index=False)
print("✅ Done! Results saved to Excel.")
