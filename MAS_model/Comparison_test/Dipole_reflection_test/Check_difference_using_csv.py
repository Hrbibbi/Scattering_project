#----------------------------------------------------------------------------
#             Comparison test for evaluation of Hertzian Dipoles
#----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import subprocess

def run_cpp_code(executable_path, param_file, testpoints_file, output_file):
    """
    Runs the C++ executable with the given parameters and testpoints file.
    Assumes the executable takes input/output file paths as command-line arguments.
    """
    try:
        subprocess.run([executable_path, param_file, testpoints_file, output_file], check=True)
        print(f"C++ implementation ran successfully, output saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ implementation: {e}")

def compare_csv_files(file1, file2):
    """
    Compares two CSV files column-wise by computing total absolute differences.
    """
    # Load the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Ensure the columns match
    if list(df1.columns) != list(df2.columns):
        raise ValueError("CSV column names do not match!")

    # Compute absolute difference for each column
    differences = {}
    for col in df1.columns:
        differences[col] = np.mean(np.abs(df1[col] - df2[col]))

    # Print differences
    print("\nTotal differences between A_simple.csv and PN_simple.csv:")
    for key, value in differences.items():
        print(f"{key}: {value}")

#----------------------------------------------------------------------------
#                           Data creation simple
#----------------------------------------------------------------------------

if False:
    mu=1
    epsilon_air=1
    epsilon_substrate=2
    omega=1
    position=[0,0,0]
    direction=[1,0,0]
    testpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # Fixed incorrect array creation

    param_data = [
    ["mu", mu],
    ["epsilon_air", epsilon_air],
    ["epsilon_substrate", epsilon_substrate],
    ["omega", omega],
    ["position_x", position[0]],
    ["position_y", position[1]],
    ["position_z", position[2]],
    ["direction_x", direction[0]],
    ["direction_y", direction[1]],
    ["direction_z", direction[2]],
    ]

    # Convert testpoints to DataFrame and add column names
    testpoints_df = pd.DataFrame(testpoints, columns=["testpoint_x", "testpoint_y", "testpoint_z"])

    # Convert parameters to DataFrame
    param_df = pd.DataFrame(param_data, columns=["Parameter", "Value"])

    # Save both to CSV
    param_df.to_csv("dipole_params_simple.csv", index=False)
    testpoints_df.to_csv("dipole_testpoints_simple.csv", index=False)

#----------------------------------------------------------------------------
#                           Data creation random
#----------------------------------------------------------------------------

if False:  
    np.random.seed(42)

    # Generate random values
    mu = np.random.uniform(0.5, 10)
    epsilon_air = np.random.uniform(0.5, 10)
    epsilon_substrate=np.random.uniform(0.5,10)
    omega = np.random.uniform(0.5, 10)

    # Random position (assuming within a range, e.g., -10 to 10)
    position = np.random.uniform(-10, 10, size=3).tolist()

    # Random unit vector for direction
    random_vector = np.random.uniform(-1, 1, size=3)
    direction = (random_vector / np.linalg.norm(random_vector)).tolist()

    # Generate 100x3 test points within a range (e.g., -10 to 10)
    testpoints = np.random.uniform(-10, 10, size=(100, 3)).tolist()

    param_data = [
    ["mu", mu],
    ["epsilon_air", epsilon_air],
    ["epsilon_substrate", epsilon_substrate],
    ["omega", omega],
    ["position_x", position[0]],
    ["position_y", position[1]],
    ["position_z", position[2]],
    ["direction_x", direction[0]],
    ["direction_y", direction[1]],
    ["direction_z", direction[2]],
    ]

    # Convert testpoints to DataFrame and add column names
    testpoints_df = pd.DataFrame(testpoints, columns=["testpoint_x", "testpoint_y", "testpoint_z"])

    # Convert parameters to DataFrame
    param_df = pd.DataFrame(param_data, columns=["Parameter", "Value"])

    # Save both to CSV
    param_df.to_csv("dipole_params_random.csv", index=False)
    testpoints_df.to_csv("dipole_testpoints_random.csv", index=False)

#----------------------------------------------------------------------------
#                         Simple case for single Dipole
#----------------------------------------------------------------------------
 
if False:
    # -----------------------------------------------
    #               A Implementation (Python)
    # -----------------------------------------------
    print("Running Python implementation...")
    import HD_comparison
    HD_comparison.compute_fields_from_csv("dipole_params_simple.csv", "dipole_testpoints_simple.csv", "A_simple.csv")

    # -----------------------------------------------
    #               P and N Implementation (C++)
    # -----------------------------------------------
    '''
    If you implement a function that takes as inout the dipole_params csv 
    and dipole_testpoints csv and then spits out the E and H fields in a csv stored as
    Ex_re,Ex_im,Ey_re,Ey_im,Ez_re,Ez_im,Hx_re,Hx_im,Hy_re,Hy_im,Hz_re,Hz_im
    '''   
    print("Running C++ implementation...")
    run_cpp_code("./PN_dipole_solver", "dipole_params_simple.csv", "dipole_testpoints_simple.csv", "PN_simple.csv")

    # -----------------------------------------------
    #               Compare Results
    # -----------------------------------------------
    compare_csv_files("A_simple.csv", "PN_simple.csv")

if False:
    # -----------------------------------------------
    #               A Implementation (Python)
    # -----------------------------------------------
    print("Running Python implementation...")
    import HD_comparison
    HD_comparison.compute_fields_from_csv("dipole_params_random.csv", "dipole_testpoints_random.csv", "A_random.csv")

    # -----------------------------------------------
    #               P and N Implementation (C++)
    # -----------------------------------------------
    '''
    If you implement a function that takes as inout the dipole_params csv 
    and dipole_testpoints csv and then spits out the E and H fields in a csv stored as
    Ex_re,Ex_im,Ey_re,Ey_im,Ez_re,Ez_im,Hx_re,Hx_im,Hy_re,Hy_im,Hz_re,Hz_im
    '''   
    print("Running C++ implementation...")
    run_cpp_code("./PN_dipole_solver", "dipole_params_random.csv", "dipole_testpoints_random.csv", "PN_random.csv")

    # -----------------------------------------------
    #               Compare Results
    # -----------------------------------------------
    compare_csv_files("A_simple.csv", "PN_simple.csv")

import HD_comparison
HD_comparison.compute_fields_from_csv("dipole_params_random.csv", "dipole_testpoints_random.csv", "A_random.csv")