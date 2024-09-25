import json
import numpy as np
import os

name = os.environ.get("name", None)
def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def check_and_merge_2d_lists(json_files):
    merged_list = None
    for file in json_files:
        data = load_json(file)[0]
        
        # Assuming the data is a 2D list
        data_np = np.array(data)

        if merged_list is None:
            # Initialize the merged list with the first file's data
            merged_list = np.zeros_like(data_np)

        # Check if any position in `merged_list` and `data_np` both are greater than 0
        conflict = (merged_list > 0) & (data_np > 0)
        if np.any(conflict):
            raise ValueError(f"Conflict detected between files at {file}")
        
        # Merge by adding non-zero values from `data_np` to `merged_list`
        merged_list = np.where(data_np > 0, data_np, merged_list)

    return merged_list.tolist()

# Example usage
if __name__ == "__main__":
    # List of JSON file paths
    from arithmetic_eval_quicker import grid_plotter
    path = f'/scratch/gpfs/pw4811/arithmetic/cramming-data/{name}/downstream'
    json_files = [
        f'{path}/accs_grid_quick_big_eval_{i}.json' for i in range(1, 9)
        ]
    try:
        merged_result = check_and_merge_2d_lists(json_files)
        print("Merged list:", merged_result)
        grid_plotter(merged_result, name=f'_{name}')
    except ValueError as e:
        print(str(e))
