import random
import os
import numpy as np

# --- Point Generation Functions (from generate_custom_sensor_locations.py) ---

def generate_poisson_process_points(area_x, area_y, num_points_to_generate):
    """Generates uniformly random points in a given area."""
    points = []
    for _ in range(num_points_to_generate):
        x = random.uniform(0, area_x)
        y = random.uniform(0, area_y)
        points.append([x, y])
    return points

def generate_gaussian_cluster_points(mu, sigma, num_points_per_cluster, area_x, area_y):
    """Generates points based on a Gaussian distribution."""
    points = []
    if isinstance(sigma, (int, float)):
        cov_matrix = [[sigma**2, 0], [0, sigma**2]]
    else:
        cov_matrix = sigma

    generated_points = np.random.multivariate_normal(mu, cov_matrix, num_points_per_cluster)
    for p in generated_points:
        x = max(0, min(p[0], area_x))
        y = max(0, min(p[1], area_y))
        points.append([x, y])
    return points

def generate_grid_points(rows, cols, spacing_x, spacing_y, area_x, area_y, jitter=0.0):
    """Generates points in a grid layout with optional jitter."""
    points = []
    start_x = (area_x - (cols - 1) * spacing_x) / 2
    start_y = (area_y - (rows - 1) * spacing_y) / 2

    for r in range(rows):
        for c in range(cols):
            x = start_x + c * spacing_x + random.uniform(-jitter, jitter)
            y = start_y + r * spacing_y + random.uniform(-jitter, jitter)
            x = max(0, min(x, area_x))
            y = max(0, min(y, area_y))
            points.append([x, y])
    return points

# --- Scenario Generation Functions (adapted to take num_points) ---

AREA_X = 500.0
AREA_Y = 500.0

def generate_forest_scenario(num_points):
    """Generates sparse random distribution."""
    return generate_poisson_process_points(AREA_X, AREA_Y, num_points)

def generate_urban_scenario(num_points):
    """Generates dense clustered distribution."""
    alpha = 0.8
    num_random_points = int((1 - alpha) * num_points)
    num_cluster_points_total = num_points - num_random_points

    points = generate_poisson_process_points(AREA_X, AREA_Y, num_random_points)

    num_clusters = 5
    if num_clusters > 0 and num_cluster_points_total > 0:
        points_per_cluster = num_cluster_points_total // num_clusters
        assigned_points = 0
        for i in range(num_clusters):
            mu_c = [random.uniform(0.1 * AREA_X, 0.9 * AREA_X), random.uniform(0.1 * AREA_Y, 0.9 * AREA_Y)]
            sigma_c = random.uniform(15, 40)
            
            num_to_gen = points_per_cluster if i < num_clusters - 1 else num_cluster_points_total - assigned_points
            if num_to_gen > 0:
                points.extend(generate_gaussian_cluster_points(mu_c, sigma_c, num_to_gen, AREA_X, AREA_Y))
                assigned_points += num_to_gen
    
    while len(points) < num_points:
        points.append(generate_poisson_process_points(AREA_X, AREA_Y, 1)[0])
    
    return points[:num_points]

def generate_agriculture_scenario(num_points):
    """Generates grid-like distribution."""
    side = int(np.sqrt(num_points))
    rows, cols = side, side
    while rows * cols < num_points:
        if rows <= cols:
            rows += 1
        else:
            cols += 1
            
    spacing_x = AREA_X / (cols + 1)
    spacing_y = AREA_Y / (rows + 1)
    
    points = generate_grid_points(rows, cols, spacing_x, spacing_y, AREA_X, AREA_Y, jitter=2.0)
    
    random.shuffle(points)
    return points[:num_points]

def generate_factory_scenario(num_points):
    """Generates structured linear distribution."""
    num_linear_structures = 3
    all_points = []
    
    if num_points == 0:
        return all_points

    raw_weights = [(c + 1)**(-2.5) for c in range(num_linear_structures)]
    z = sum(raw_weights)
    normalized_weights = [w / z for w in raw_weights]
    
    assigned_points = 0
    for i in range(num_linear_structures):
        num_to_gen = int(normalized_weights[i] * num_points) if i < num_linear_structures - 1 else num_points - assigned_points
        if num_to_gen <= 0:
            continue

        is_horizontal = random.choice([True, False])
        if is_horizontal:
            mu_y, mu_x = random.uniform(0.1 * AREA_Y, 0.9 * AREA_Y), random.uniform(0.2 * AREA_X, 0.8 * AREA_X)
            sigma_x_sq, sigma_y_sq = random.uniform(50, 100)**2, random.uniform(5, 15)**2
        else:
            mu_x, mu_y = random.uniform(0.1 * AREA_X, 0.9 * AREA_X), random.uniform(0.2 * AREA_Y, 0.8 * AREA_Y)
            sigma_x_sq, sigma_y_sq = random.uniform(5, 15)**2, random.uniform(50, 100)**2
        
        cov = [[sigma_x_sq, 0], [0, sigma_y_sq]]
        all_points.extend(generate_gaussian_cluster_points([mu_x, mu_y], cov, num_to_gen, AREA_X, AREA_Y))
        assigned_points += num_to_gen

    while len(all_points) < num_points:
        all_points.append(generate_poisson_process_points(AREA_X, AREA_Y, 1)[0])
    
    return all_points[:num_points]

def generate_custom_hybrid_scenario(num_points):
    """Generates a mix of random, linear, and clustered points."""
    points = []
    
    num_random = int(0.2 * num_points)
    num_linear = int(0.5 * num_points)
    num_dense = num_points - num_random - num_linear

    points.extend(generate_poisson_process_points(AREA_X, AREA_Y, num_random))

    if num_linear > 0:
        angle = random.uniform(0, np.pi)
        length = random.uniform(AREA_X * 0.4, AREA_X * 0.8)
        center = [random.uniform(length / 2, AREA_X - length / 2), random.uniform(length / 2, AREA_Y - length / 2)]
        sigma_major_sq, sigma_minor_sq = (length / 4)**2, random.uniform(5, 15)**2
        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        D = np.array([[sigma_major_sq, 0], [0, sigma_minor_sq]])
        cov = R @ D @ R.T
        points.extend(generate_gaussian_cluster_points(center, cov, num_linear, AREA_X, AREA_Y))

    if num_dense > 0:
        mu = [random.uniform(0.1 * AREA_X, 0.9 * AREA_X), random.uniform(0.1 * AREA_Y, 0.9 * AREA_Y)]
        sigma = random.uniform(5, 15)
        points.extend(generate_gaussian_cluster_points(mu, sigma, num_dense, AREA_X, AREA_Y))

    while len(points) < num_points:
        points.append(generate_poisson_process_points(AREA_X, AREA_Y, 1)[0])
        
    return points[:num_points]

# --- Main TSP File Creation Logic ---

SCENARIO_GENERATORS = {
    "forest": generate_forest_scenario,
    "urban": generate_urban_scenario,
    "agriculture": generate_agriculture_scenario,
    "factory": generate_factory_scenario,
    "custom_hybrid": generate_custom_hybrid_scenario
}

def create_tsp_file(scenario_name, output_directory, nodes_to_select, instance_num):
    """Generates sensor locations based on a scenario and creates a TSP file."""
    if scenario_name not in SCENARIO_GENERATORS:
        print(f"Error: Unknown scenario '{scenario_name}'. Skipping.")
        return

    num_sensor_nodes = nodes_to_select - 1
    
    generator_func = SCENARIO_GENERATORS[scenario_name]
    sensor_points = generator_func(num_sensor_nodes)

    depot_point = [AREA_X / 2, AREA_Y / 2]
    all_coords = [depot_point] + sensor_points

    tsp_content = f"NAME: {scenario_name}_{nodes_to_select}_{instance_num}\n"
    tsp_content += "TYPE: TSP\n"
    tsp_content += f"COMMENT: {nodes_to_select} nodes for {scenario_name}, instance {instance_num}\n"
    tsp_content += f"DIMENSION: {nodes_to_select}\n"
    tsp_content += "EDGE_WEIGHT_TYPE: EUC_2D\n"
    tsp_content += "NODE_COORD_SECTION\n"

    for i, coords in enumerate(all_coords):
        tsp_content += f"{i + 1} {coords[0]:.4f} {coords[1]:.4f}\n"

    tsp_content += "EOF\n"

    output_filename = f"sensor_locations_{scenario_name}_{nodes_to_select}_{instance_num}.tsp"
    output_path = os.path.join(output_directory, output_filename)
    with open(output_path, 'w') as f:
        f.write(tsp_content)
    print(f"Created TSP file: {output_path}")


if __name__ == '__main__':
    # --- Configuration ---
    output_dir = "showcases/Simulating_a_data_collection_scenario_static/data/"
    
    dataset_sizes = [50, 100, 150, 200]
    scenarios = ["agriculture", "custom_hybrid", "factory", "forest", "urban"]
    num_instances_per_scenario = 10

    # --- Execution ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    for size in dataset_sizes:
        print(f"\n--- Generating datasets for size: {size} ---")
        for scenario in scenarios:
            print(f"  Processing scenario: {scenario}...")
            for i in range(num_instances_per_scenario):
                create_tsp_file(
                    scenario_name=scenario,
                    output_directory=output_dir,
                    nodes_to_select=size,
                    instance_num=i + 1
                )
    
    print("\n--- Dataset generation complete. ---")