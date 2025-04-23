import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def generate_grid(width=1, resol=20):
    x = np.linspace(-width, width, resol)
    y = np.linspace(-width, width, resol)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    return x, y, X, Y, Z

def add_random_bump(X, Y, Z, bump_data_list):
    x_min, x_max = np.min(X), np.max(X)
    y_min, y_max = np.min(Y), np.max(Y)

    x0 = np.random.uniform(x_min + 0.1*(x_max - x_min), x_max - 0.1*(x_max - x_min))
    y0 = np.random.uniform(y_min + 0.1*(y_max - y_min), y_max - 0.1*(y_max - y_min))
    height = np.random.uniform(0.1, 0.4)
    sigma = np.random.uniform(0.2, 0.7)

    bump = height * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    Z_new = Z + bump

    bump_data = {
        "x0": x0,
        "y0": y0,
        "height": height,
        "sigma": sigma
    }
    bump_data_list.append(bump_data)

    return Z_new, bump_data_list

def export_bump_data(bump_data, path):
    bump_df = pd.DataFrame(bump_data)
    bump_df.to_csv(path, index=False)

def export_surface_full(X, Y, Z, path):
    df = pd.DataFrame({
        'x': X.ravel(),
        'y': Y.ravel(),
        'z': Z.ravel()
    })
    df.to_csv(path, index=False)

def export_MAS_data(data, path):
    # --- Export test points with normals and tangents ---
    df_test = pd.DataFrame({
        "x": data["test_points"][:, 0],
        "y": data["test_points"][:, 1],
        "z": data["test_points"][:, 2],
        "nx": data["test_normals"][:, 0],
        "ny": data["test_normals"][:, 1],
        "nz": data["test_normals"][:, 2],
        "t1x": data["tangent1"][:, 0],
        "t1y": data["tangent1"][:, 1],
        "t1z": data["tangent1"][:, 2],
        "t2x": data["tangent2"][:, 0],
        "t2y": data["tangent2"][:, 1],
        "t2z": data["tangent2"][:, 2]
    })
    df_test.to_csv(path + "/test_points.csv", index=False)

    # --- Export auxiliary points with matching test point index ---
    aux_idx = data["aux_indices"]

    df_aux = pd.DataFrame({
        "type": ["int"] * len(aux_idx) + ["ext"] * len(aux_idx),
        "x": np.concatenate([data["aux_points_int"][:, 0], data["aux_points_ext"][:, 0]]),
        "y": np.concatenate([data["aux_points_int"][:, 1], data["aux_points_ext"][:, 1]]),
        "z": np.concatenate([data["aux_points_int"][:, 2], data["aux_points_ext"][:, 2]]),
        "test_index": np.concatenate([aux_idx, aux_idx])
    })
    df_aux.to_csv(path + "/aux_points.csv", index=False)

    print("✅ MAS data exported to test_points.csv and aux_points.csv")

def plot_surface_contour(X, Y, Z):
    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(contour, label="Surface height z (μm)")
    plt.title("Random Gaussian Bump Surface")
    plt.xlabel("x (μm)")
    plt.ylabel("y (μm)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

def approx_peak_curvature(Z, x, y):
    peak_idx = np.unravel_index(np.argmax(Z), Z.shape)
    dz_dx2 = np.gradient(np.gradient(Z, x, axis=1), x, axis=1)
    dz_dy2 = np.gradient(np.gradient(Z, y, axis=0), y, axis=0)
    curv_xx = dz_dx2[peak_idx]
    curv_yy = dz_dy2[peak_idx]
    mean_curv = 0.5 * (curv_xx + curv_yy)
    return mean_curv

def generate_MAS_model(x, y, X, Y, Z, lambda_0=0.325, d = 0.14, testPointPerLambda = 5):
    Lx = x[-1] - x[0]
    Ly = y[-1] - y[0]
    N_test = int(np.ceil(2 * (testPointPerLambda * Lx * testPointPerLambda * Ly) / lambda_0**2))

    # --- Step 3: Flatten grid and remove edge points ---
    X_flat, Y_flat, Z_flat = X.ravel(), Y.ravel(), Z.ravel()
    points = np.stack([X_flat, Y_flat, Z_flat], axis=1)
    mask = (X_flat > x[0]) & (X_flat < x[-1]) & (Y_flat > y[0]) & (Y_flat < y[-1])
    interior_points = points[mask]

    total_pts = interior_points.shape[0]

    # --- Step 4: Sample test points ---
    idx_test = np.arange(0, total_pts, 2)  # ✅ every second point in order
    test_points = interior_points[idx_test] # ✅ preserves structure

    # After selecting test_points
    N_test = len(test_points)  # ✅ actual number used

    # For aux indices
    aux_test_indices = np.arange(0, N_test, 2)  # ✅ safe indexing

    # --- Step 5: Compute normals and tangents ---
    dz_dx, dz_dy = np.gradient(Z, x, y)
    dz_dx_flat = dz_dx.ravel()[mask]
    dz_dy_flat = dz_dy.ravel()[mask]

    normals = np.stack([-dz_dx_flat, -dz_dy_flat, np.ones_like(dz_dx_flat)], axis=1)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    test_normals = normals[idx_test]

    # Compute tangent vectors (Gram-Schmidt)
    tangent1 = np.zeros_like(test_normals)
    tangent2 = np.zeros_like(test_normals)

    for i, n in enumerate(test_normals):
        if abs(n[2]) < 0.9:
            ref = np.array([0, 0, 1])
        else:
            ref = np.array([1, 0, 0])

        t1 = ref - np.dot(ref, n) * n
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        tangent1[i] = t1
        tangent2[i] = t2

    # --- Step 6: Compute auxiliary points and mapping indices ---
    aux_test_indices = np.arange(N_test)[::2]
    aux_test_points = test_points[aux_test_indices]
    aux_test_normals = test_normals[aux_test_indices]

    aux_points_int = aux_test_points - d * aux_test_normals
    aux_points_ext = aux_test_points + d * aux_test_normals

    # --- Return full structured data ---
    return {
    "test_points": test_points,
    "test_normals": test_normals,
    "tangent1": tangent1,
    "tangent2": tangent2,
    "aux_points_int": aux_points_int,
    "aux_points_ext": aux_points_ext,
    "aux_indices": aux_test_indices,
    }

def visualize_surface_and_points(X, Y, Z, mas_data, path_prefix):
    def plot_points(points, color, name, filename):
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6))
        fig.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                   mode='markers', marker=dict(size=2, color=color), name=name))
        fig.update_layout(scene=dict(xaxis_title='x (μm)', yaxis_title='y (μm)', zaxis_title='z (μm)', aspectmode='data'),
                          title=f"3D Surface with {name}")
        fig.write_html(filename)
        print(f"✅ Saved: {filename}")

    # Plot all combined
    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6))
    fig.add_trace(go.Scatter3d(x=mas_data["test_points"][:, 0], y=mas_data["test_points"][:, 1], z=mas_data["test_points"][:, 2],
                               mode='markers', marker=dict(size=2, color='red'), name="Test points"))
    fig.add_trace(go.Scatter3d(x=mas_data["aux_points_int"][:, 0], y=mas_data["aux_points_int"][:, 1], z=mas_data["aux_points_int"][:, 2],
                               mode='markers', marker=dict(size=2, color='orange'), name="Aux (int)"))
    fig.add_trace(go.Scatter3d(x=mas_data["aux_points_ext"][:, 0], y=mas_data["aux_points_ext"][:, 1], z=mas_data["aux_points_ext"][:, 2],
                               mode='markers', marker=dict(size=2, color='cyan'), name="Aux (ext)"))
    fig.update_layout(scene=dict(xaxis_title='x (μm)', yaxis_title='y (μm)', zaxis_title='z (μm)', aspectmode='data'),
                      title="Interactive 3D Surface with Points")
    fig.write_html(path_prefix + "/surface_with_points.html")

    # Individual sets
    plot_points(mas_data["test_points"], 'red', "Test Points", path_prefix + "/surface_test_points.html")
    plot_points(mas_data["aux_points_int"], 'orange', "Auxiliary Interior Points", path_prefix + "/surface_aux_int.html")
    plot_points(mas_data["aux_points_ext"], 'cyan', "Auxiliary Exterior Points", path_prefix + "/surface_aux_ext.html")

    # Test points with and without aux
    mask_with_aux = np.zeros(len(mas_data["test_points"]), dtype=bool)
    mask_with_aux[mas_data["aux_indices"]] = True
    test_with_aux = mas_data["test_points"][mask_with_aux]
    test_without_aux = mas_data["test_points"][~mask_with_aux]

    fig = go.Figure()
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.6))
    fig.add_trace(go.Scatter3d(x=test_with_aux[:, 0], y=test_with_aux[:, 1], z=test_with_aux[:, 2],
                               mode='markers', marker=dict(size=3, color='blue'), name="Test points with aux"))
    fig.add_trace(go.Scatter3d(x=test_without_aux[:, 0], y=test_without_aux[:, 1], z=test_without_aux[:, 2],
                               mode='markers', marker=dict(size=2, color='gray'), name="Other test points"))
    fig.update_layout(scene=dict(xaxis_title='x (μm)', yaxis_title='y (μm)', zaxis_title='z (μm)', aspectmode='data'),
                      title="Test Points With vs Without Aux Points")
    fig.write_html(path_prefix + "/test_points_with_vs_without_aux.html")
    print("✅ Saved: ./SurfaceData/test_points_with_vs_without_aux.html")



def main():
    np.random.seed(42)
    width = 1
    resol = 20
    num_bumps = 3
    data_path = "./SurfaceData"
    import os
    os.makedirs(data_path, exist_ok=True)

    x, y, X, Y, Z = generate_grid(width, resol)
    bump_data = []

    for _ in range(num_bumps):
        Z, bump_data = add_random_bump(X, Y, Z, bump_data)

    export_bump_data(bump_data, data_path + "/bumpData.csv")

    export_surface_full(X, Y, Z, data_path + "/SurfaceData.csv")

    plot_surface_contour(X, Y, Z)

    mean_curv = approx_peak_curvature(Z, x, y)
    print(f"Mean curvature at peak: {mean_curv:.3f}")

    radius = 1 / abs(mean_curv)

    d = 0.14 * radius

    print(f"Peak curvature (mean): {mean_curv:.6f} μm⁻¹")
    print(f"Radius of curvature: r = {radius:.6f} μm")
    print(f"Final offset d = {d:.6f} μm")

    data = generate_MAS_model(x, y, X, Y, Z, lambda_0=0.325, d = 0.14, testPointPerLambda = 5)

    # Access results
    test_pts = data["test_points"]
    normals = data["test_normals"]
    t1 = data["tangent1"]
    t2 = data["tangent2"]
    aux_int = data["aux_points_int"]
    aux_ext = data["aux_points_ext"]
    aux_test_idx = data["aux_indices"]

    export_MAS_data(data, data_path)

    visualize_surface_and_points(X, Y, Z, data, data_path)

if __name__ == "__main__":
    main()
