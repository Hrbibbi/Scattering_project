import numpy as np
import plotly.graph_objects as go
import os
from scipy.spatial import cKDTree


def plot_dipole_orientations(points, alpha_tensor, output_folder, file_name, percentage=100):
    assert 0 < percentage <= 100, "Percentage must be in (0, 100]."

    M = points.shape[0]
    z_hat = np.array([0, 0, 1])
    directions = np.einsum('mij,j->mi', alpha_tensor, z_hat)

    num_to_plot = int(M * (percentage / 100.0))
    indices = np.random.choice(M, num_to_plot, replace=False)

    pts = points[indices]
    dirs = directions[indices]
    dirs_norm = dirs / np.linalg.norm(dirs, axis=1, keepdims=True) * 5

    fig = go.Figure()
    for i in range(num_to_plot):
        x0, y0, z0 = pts[i]
        u, v, w = dirs_norm[i]
        fig.add_trace(go.Scatter3d(
            x=[x0, x0 + u],
            y=[y0, y0 + v],
            z=[z0, z0 + w],
            mode="lines+markers",
            line=dict(color="blue", width=4),
            marker=dict(size=2),
            showlegend=False
        ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=f'Dipole Orientations ({percentage:.1f}%)',
        margin=dict(l=0, r=0, b=0, t=30)
    )

    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, file_name + ".html")
    fig.write_html(file_path)
    print(f"Saved dipole plot to: {file_path}")


def plot_euler_colorcode(points, euler_angles, output_folder, filename_prefix="euler_plot"):
    """
    Create three 3D scatter plots (yaw, pitch, roll), where each point is color-coded
    by the corresponding angle value.

    Parameters
    ----------
    points : (N, 3) array_like
        The 3D coordinates of each point, as [[x1,y1,z1], [x2,y2,z2], …].
    euler_angles : (N, 3) array_like
        The Euler angles at each point, in columns [yaw, pitch, roll].
    output_folder : str
        Directory where HTML files will be saved (created if needed).
    filename_prefix : str, optional
        Base name for output files; “_yaw.html”, “_pitch.html”, and “_roll.html”
        will be appended automatically.

    Produces
    -------
    Three HTML files in `output_folder`:
      • {filename_prefix}_yaw.html  
      • {filename_prefix}_pitch.html  
      • {filename_prefix}_roll.html
    """

    # Validate shapes
    points = np.asarray(points, dtype=float)
    euler_angles = np.asarray(euler_angles, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("`points` must be shape (N, 3).")
    if euler_angles.ndim != 2 or euler_angles.shape[1] != 3 or euler_angles.shape[0] != points.shape[0]:
        raise ValueError("`euler_angles` must be shape (N, 3), matching points.")

    # Create output directory if it doesn’t exist
    os.makedirs(output_folder, exist_ok=True)

    # Unpack coordinates
    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    # Column indices for [yaw, pitch, roll]
    labels = ["yaw", "pitch", "roll"]

    for idx, label in enumerate(labels):
        angles = euler_angles[:, idx]  # shape (N,)

        fig = go.Figure(
            go.Scatter3d(
                x = X,
                y = Y,
                z = Z,
                mode = "markers",
                marker = dict(
                    size = 4,
                    color = angles,
                    colorscale = "Viridis",
                    colorbar = dict(title=label),
                    showscale = True
                )
            )
        )

        fig.update_layout(
            scene = dict(
                xaxis_title = "X",
                yaxis_title = "Y",
                zaxis_title = "Z",
                aspectmode = "data"
            ),
            title = f"Points color-coded by {label}",
            margin = dict(l=0, r=0, b=0, t=30)
        )

        outfile = os.path.join(output_folder, f"{filename_prefix}_{label}.html")
        fig.write_html(outfile)
        print(f"Saved {label} plot to: {outfile}")


def plot_streamlines(points,alpha_tensor,voxel_size, output_path="streamlines.html"):

    def trace_streamlines(points, directions, voxel_size, max_steps=100):
        # Build KD-tree for fast membership check
        tree = cKDTree(points)
        point_set = set(map(tuple, np.round(points, 5)))

        # Find base points: z == 0
        base_mask = np.abs(points[:, 2]) < 1e-5
        base_points = points[base_mask]
        base_directions = directions[base_mask]

        streamlines = []

        for start_pt, dir0 in zip(base_points, base_directions):
            streamline = [start_pt]
            current_pt = start_pt.copy()
            current_dir = dir0 / np.linalg.norm(dir0)

            for _ in range(max_steps):
                next_pt = current_pt + voxel_size * current_dir
                key = tuple(np.round(next_pt, 5))

                if key not in point_set:
                    break  # we've exited the ellipsoid

                streamline.append(next_pt)
                current_pt = next_pt

                # Update direction at the new point
                dist, idx = tree.query(current_pt)
                current_dir = directions[idx]
                current_dir = current_dir / np.linalg.norm(current_dir)

            streamlines.append(np.array(streamline))

        return streamlines
    z_hat=np.array([0,0,1]) 
    directions=np.einsum('mij,j->mi', alpha_tensor, z_hat)
    streamlines=trace_streamlines(points,directions,voxel_size)
    fig = go.Figure()

    for line in streamlines:
        fig.add_trace(go.Scatter3d(
            x=line[:, 0], y=line[:, 1], z=line[:, 2],
            mode='lines',
            line=dict(width=3),
            opacity=0.8
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode='data'
        ),
        title="Streamlines from Base Plane Through Protrusion",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    fig.write_html(output_path)
    print(f"Saved streamline plot to: {output_path}")
