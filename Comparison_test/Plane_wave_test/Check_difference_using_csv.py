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
    epsilon=1
    omega=1
    propagation_vector=[0,-1,0]
    polarization=0
    testpoints = np.array([[0,1,0],[0,2,0],[0,3,0]])
    param_data = [
    ["mu", mu],
    ["epsilon", epsilon],
    ["omega", omega],
    ['polarization',polarization],
    ["propagation_x", propagation_vector[0]],
    ["propagation_y", propagation_vector[1]],
    ["propagation_z", propagation_vector[2]],
    ]

    # Convert testpoints to DataFrame and add column names
    testpoints_df = pd.DataFrame(testpoints, columns=["testpoint_x", "testpoint_y", "testpoint_z"])

    # Convert parameters to DataFrame
    param_df = pd.DataFrame(param_data, columns=["Parameter", "Value"])

    # Save both to CSV
    param_df.to_csv("PW_params_simple.csv", index=False)
    testpoints_df.to_csv("PW_testpoints_simple.csv", index=False)

#----------------------------------------------------------------------------
#                           Data creation random
#----------------------------------------------------------------------------

if False:  
    np.random.seed(42)

    # Generate random values
    mu = np.random.uniform(0.5, 10)
    epsilon = np.random.uniform(0.5, 10)
    omega = np.random.uniform(0.5, 10)
    polarization = np.random.uniform(0,np.pi/2)
    # Random unit vector for direction
    random_vector = np.random.uniform(-1, 1, size=3)
    propagation_vector = (random_vector / np.linalg.norm(random_vector)).tolist()

    # Generate 100x3 test points within a range (e.g., -10 to 10)
    testpoints = np.random.uniform(-10, 10, size=(100, 3)).tolist()

    param_data = [
    ["mu", mu],
    ["epsilon", epsilon],
    ["omega", omega],
    ['polarization',polarization],
    ["propagation_x", propagation_vector[0]],
    ["propagation_y", propagation_vector[1]],
    ["propagation_z", propagation_vector[2]],
    ]

    # Convert testpoints to DataFrame and add column names
    testpoints_df = pd.DataFrame(testpoints, columns=["testpoint_x", "testpoint_y", "testpoint_z"])

    # Convert parameters to DataFrame
    param_df = pd.DataFrame(param_data, columns=["Parameter", "Value"])

    # Save both to CSV
    param_df.to_csv("PW_params_random.csv", index=False)
    testpoints_df.to_csv("PW_testpoints_random.csv", index=False)

#----------------------------------------------------------------------------
#                         Simple case for single Dipole
#----------------------------------------------------------------------------
 
if False:
    # -----------------------------------------------
    #               A Implementation (Python)
    # -----------------------------------------------
    print("Running Python implementation...")
    import plane_wave_comparison
    plane_wave_comparison.compute_fields_from_csv("PW_params_simple.csv","PW_testpoints_simple.csv","A_simple.csv")

    # -----------------------------------------------
    #               P and N Implementation (C++)
    # -----------------------------------------------
    '''
    If you implement a function that takes as inout the dipole_params csv 
    and dipole_testpoints csv and then spits out the E and H fields in a csv stored as
    Ex_re,Ex_im,Ey_re,Ey_im,Ez_re,Ez_im,Hx_re,Hx_im,Hy_re,Hy_im,Hz_re,Hz_im
    '''   
    print("Running C++ implementation...")
    run_cpp_code("./PN_dipole_solver", "PW_params_simple.csv","PW_testpoints_simple.csv","A_simple.csv")

    # -----------------------------------------------
    #               Compare Results
    # -----------------------------------------------
    compare_csv_files("A_simple.csv", "PN_simple.csv")

if False:
    # -----------------------------------------------
    #               A Implementation (Python)
    # -----------------------------------------------
    print("Running Python implementation...")
    import plane_wave_comparison
    plane_wave_comparison.compute_fields_from_csv("PW_params_random.csv","PW_testpoints_random.csv","A_random.csv")

    # -----------------------------------------------
    #               P and N Implementation (C++)
    # -----------------------------------------------
    '''
    If you implement a function that takes as inout the dipole_params csv 
    and dipole_testpoints csv and then spits out the E and H fields in a csv stored as
    Ex_re,Ex_im,Ey_re,Ey_im,Ez_re,Ez_im,Hx_re,Hx_im,Hy_re,Hy_im,Hz_re,Hz_im
    '''   
    print("Running C++ implementation...")
    run_cpp_code("./PN_dipole_solver", "PW_params_random.csv","PW_testpoints_random.csv", "PN_random.csv")

    # -----------------------------------------------
    #               Compare Results
    # -----------------------------------------------
    compare_csv_files("A_simple.csv", "PN_simple.csv")
