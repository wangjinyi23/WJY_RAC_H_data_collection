import json
import random
import os
import subprocess
import tempfile
import time
import logging
from typing import List, Tuple, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

Position3D = Tuple[float, float, float]

def load_sensor_locations(file_path: str) -> Dict[int, Position3D]:
    """Loads sensor locations from a file."""
    with open(file_path, 'r') as f:
        return {int(k): tuple(v) for k, v in json.load(f).items()}

def write_tsp_file(filename: str, points: List[Position3D], name="TSP", depot_node=1):
    """Writes a TSP problem file."""
    with open(filename, 'w') as f:
        f.write(f"NAME: {name}\n")
        f.write("TYPE: TSP\n")
        f.write(f"DIMENSION: {len(points)}\n")
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, (x, y, z) in enumerate(points):
            f.write(f"{i + 1} {x} {y}\n")
        f.write(f"DEPOT_SECTION\n{depot_node}\n-1\n")
        f.write("EOF\n")

def solve_tsp_with_lkh(locations_to_visit: List[Position3D]) -> Tuple[Optional[List[Position3D]], float]:
    """Solves TSP using LKH and returns the tour and execution time."""
    if len(locations_to_visit) <= 1:
        return [], 0.0

    lkh_path = os.path.abspath("showcases/Simulating_a_data_collection_scenario/LKH-3.0.13/LKH")
    if not os.path.exists(lkh_path):
        logging.error(f"LKH executable not found at {lkh_path}")
        return None, 0.0

    with tempfile.TemporaryDirectory() as tmpdir:
        problem_file = os.path.join(tmpdir, "problem.tsp")
        par_file = os.path.join(tmpdir, "problem.par")
        tour_file = os.path.join(tmpdir, "solution.tour")

        write_tsp_file(problem_file, locations_to_visit, name="LKH_Analysis_TSP", depot_node=1)

        par_content = f"""PROBLEM_FILE = {problem_file}
TOUR_FILE = {tour_file}
MOVE_TYPE = 5
PATCHING_C = 3
PATCHING_A = 2
RUNS = 1
"""
        with open(par_file, 'w') as f:
            f.write(par_content)

        try:
            start_time = time.time()
            subprocess.run([lkh_path, par_file], check=True, cwd=tmpdir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            end_time = time.time()
            
            with open(tour_file, 'r') as f:
                in_tour_section = False
                tour_indices = []
                for line in f:
                    line = line.strip()
                    if line == "TOUR_SECTION":
                        in_tour_section = True
                        continue
                    if line == "-1" or line == "EOF":
                        break
                    if in_tour_section:
                        node_id = int(line)
                        if node_id != 1:  # Exclude depot from the start of the tour
                            tour_indices.append(node_id)
            
            tour = [locations_to_visit[i - 1] for i in tour_indices]
            return tour, end_time - start_time

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.error(f"LKH solver failed: {e}")
            return None, 0.0

def main():
    """Main function to run the analysis."""
    scenario_files = [
        "showcases/Simulating_a_data_collection_scenario/sensor_locations_agriculture.json",
        "showcases/Simulating_a_data_collection_scenario/sensor_locations_custom_hybrid.json",
        "showcases/Simulating_a_data_collection_scenario/sensor_locations_factory.json",
        "showcases/Simulating_a_data_collection_scenario/sensor_locations_forest.json",
        "showcases/Simulating_a_data_collection_scenario/sensor_locations_urban.json",
    ]
    num_nodes_to_select = 200
    results = {}

    for file_path in scenario_files:
        if not os.path.exists(file_path):
            logging.warning(f"Scenario file not found: {file_path}")
            continue

        scenario_name = os.path.basename(file_path).replace("sensor_locations_", "").replace(".json", "")
        logging.info(f"--- Processing Scenario: {scenario_name} ---")

        all_locations = load_sensor_locations(file_path)
        
        # Ground station is node 0
        ground_station_pos = all_locations.get(0)
        if ground_station_pos is None:
            logging.error(f"Ground station (node 0) not found in {file_path}")
            continue

        sensor_nodes = {k: v for k, v in all_locations.items() if k != 0}
        
        if len(sensor_nodes) < num_nodes_to_select:
            logging.warning(f"Scenario '{scenario_name}' has fewer than {num_nodes_to_select} sensors. Using all {len(sensor_nodes)} sensors.")
            selected_sensor_keys = list(sensor_nodes.keys())
        else:
            selected_sensor_keys = random.sample(list(sensor_nodes.keys()), num_nodes_to_select)

        selected_locations = [sensor_nodes[key] for key in selected_sensor_keys]
        
        # The depot is always the first location for LKH
        locations_to_visit = [ground_station_pos] + selected_locations

        logging.info(f"Running LKH for {len(locations_to_visit)} nodes...")
        tour, exec_time = solve_tsp_with_lkh(locations_to_visit)

        if tour is not None:
            logging.info(f"LKH execution time: {exec_time:.4f} seconds")
            results[scenario_name] = exec_time
        else:
            logging.error("LKH failed to produce a tour.")
            results[scenario_name] = "Failed"

    print("\n--- LKH Performance Analysis Results ---")
    for scenario, timing in results.items():
        if isinstance(timing, float):
            print(f"Scenario: {scenario.ljust(20)} | Execution Time: {timing:.4f} seconds")
        else:
            print(f"Scenario: {scenario.ljust(20)} | Status: {timing}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()