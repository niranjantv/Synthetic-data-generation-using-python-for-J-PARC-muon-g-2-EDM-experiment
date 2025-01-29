import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
import random

# Constants

m_e = 0.511  # Rest mass of positron (MeV/c^2)
c = 3e8  # Speed of light (m/s)

# Detector geometry parameters  
r_min = 90  # Minimum radius (mm)
r_max = 290  # Maximum radius (mm)
z_min = -200  # Minimum height (mm)
z_max = 200  # Maximum height (mm)
num_vanes = 40  # Number of radial vanes
vane_thickness = 0.32  # Vane thickness (mm)

# Magnetic field
B = 3  # Magnetic field strength (T)


# Calculate delta_phi globally
delta_phi = vane_thickness / r_min  # Angular width of each vane (radians)


# Positron momentum range for analysis 
momentum_min = 30  # Minimum momentum (MeV/c)
momentum_max = 290  # Maximum momentum (MeV/c)

# Detector efficiency (e.g., 90% efficiency means 10% chance of inefficiency)
detector_efficiency = 0.9

# Secondary electron generation probability
secondary_probability = 1  # Probability of generating secondary electrons


def calculate_velocity(momentum):
    """
    Calculate the relativistic velocity of a positron given its momentum.

    Args:
        momentum (float): Momentum of the positron in MeV/c.

    Returns:
        float: Velocity of the positron in mm/ns.
    """
    energy = np.sqrt((momentum ** 2) + (m_e ** 2))  # Total energy in MeV
    velocity = (momentum / energy) * c * 1e-6  # Velocity in mm/ns (corrected)
    return velocity


def get_nearest_timestamp(time_ns):
    """
    Round the given time to the nearest 5 ns interval.

    Args:
        time_ns (float): Time in nanoseconds.

    Returns:
        float: Nearest 5 ns interval.
    """
    return 5 * np.ceil(time_ns / 5)


def should_record_hit():
    """
    Determine whether to record a hit based on detector efficiency.

    Returns:
        bool: True if the hit should be recorded, False otherwise.
    """
    return np.random.rand() < detector_efficiency  # Random check


def add_noise_points(hit_points, num_hit_points_with_noise, num_noise_points, noise_radius):
    """
    Add noise points around existing hit points.

    Args:
        hit_points (list): List of hit points.
        num_hit_points_with_noise (int): Number of hit points to add noise to.
        num_noise_points (int): Number of noise points to add per hit point.
        noise_radius (float): Standard deviation of the noise distribution.

    Returns:
        list: Updated list of hit points with noise added.
    """
    new_hit_points = []

    # Randomly choose which hit points will receive noise
    selected_hit_points = random.sample(hit_points, min(num_hit_points_with_noise, len(hit_points)))

    for hit in hit_points:
        x, y, z, timestamp, time_accumulated = hit
        new_hit_points.append(hit)  # Add the original hit point

        if hit in selected_hit_points:  # Add noise for selected hit points
            for _ in range(num_noise_points):
                noise_x, noise_y, noise_z = np.random.normal(0, noise_radius, 3)
                new_hit_points.append([x + noise_x, y + noise_y, z + noise_z, timestamp, time_accumulated])

    return new_hit_points


def add_secondary_electrons_to_random_hit_points(
    hit_points, 
    num_hit_points_range=(0, 5), 
    num_secondary_range=(5, 10), 
    z_variation=10
):
    """
    Add secondary electrons to random hit points.

    Args:
        hit_points (list): List of hit points.
        num_hit_points_range (tuple): Range of hit points to add secondary electrons to.
        num_secondary_range (tuple): Range of secondary electrons to add per hit point.
        z_variation (float): Maximum variation in the z-coordinate for secondary electrons.

    Returns:
        list: Updated list of hit points with secondary electrons added.
    """
    if not hit_points:  # Check if hit_points is empty
        return hit_points  # Return unchanged list to avoid error

    # Ensure we do not select more hit points than available
    num_hit_points = min(np.random.randint(*num_hit_points_range), len(hit_points))

    if num_hit_points == 0:  # Extra safety check
        return hit_points

    selected_indices = np.random.choice(len(hit_points), size=num_hit_points, replace=False)
    new_secondary_electrons = []

    for index in selected_indices:
        x, y, z, timestamp, time_accumulated = hit_points[index]

        # Determine the number of secondary electrons for this hit point
        num_secondary = np.random.randint(*num_secondary_range)

        for _ in range(num_secondary):
            # Apply a random z-offset for secondary electrons
            z_offset = np.random.uniform(-z_variation, z_variation)
            z_secondary = z + z_offset

            # Append new secondary electron hit point
            new_secondary_electrons.append([x, y, z_secondary, timestamp, time_accumulated])

    # Extend original hit points with new secondary electrons
    hit_points.extend(new_secondary_electrons)
    return hit_points


def helix(t, d0, z0, phi, kappa, tan_theta, alpha=0):
    """
    Calculate the position of a particle in a helical trajectory.

    Args:
        t (float): Parameter along the trajectory.
        d0 (float): Impact parameter (mm).
        z0 (float): Initial z-position (mm).
        phi (float): Initial angle (radians).
        kappa (float): Curvature (1/mm).
        tan_theta (float): Tangent of the theta angle.
        alpha (float): Spiral rate (default: 0).

    Returns:
        tuple: (x, y, z, R) coordinates of the particle.
    """
    R0 = 1 / kappa  # Initial radius of curvature (mm)
    z = z0 + (R0 * tan_theta) * t
    R = R0 + alpha * z  # Adjust radius based on z (inward spiral for negative alpha)

    # Calculate x and y coordinates
    x = (d0 - R) * np.cos(phi + t) + R * np.cos(phi)
    y = (d0 - R) * np.sin(phi + t) + R * np.sin(phi)

    return x, y, z, R


def generate_single_track(track_no,
     start_time,
     num_hit_points_with_noise=np.random.randint(2, 6),
     num_noise_points=np.random.randint(1,3),
     noise_radius=3.0,
     spiral_rate=np.random.uniform(-0.1, -0.5)
):
    """
    Generate a single positron track with hit points.

    Args:
        track_no (int): Track number.
        start_time (float): Start time of the track.
        num_hit_points_with_noise (int): Number of hit points to add noise to.
        num_noise_points (int): Number of noise points to add per hit point.
        noise_radius (float): Standard deviation of the noise distribution.
        spiral_rate (float): Rate of inward spiraling.

    Returns:
        tuple: (hit_points, first_hit) where hit_points is a list of hit points and first_hit is the first hit point.
    """
    # Generate random initial parameters
    p = np.random.uniform(momentum_min, momentum_max)  # Initial positron momentum (MeV/c)
    velocity = calculate_velocity(p)  # Initial velocity in mm/ns

    q = 1  # Charge of positron
    p_T = p  # Transverse momentum
    radius = p_T / (q * B)  # Initial radius of curvature (mm)
    kappa = 1 / radius  # Initial curvature

    d0 = np.random.uniform(250, 300)  # Impact parameter (mm)
    z0 = np.random.uniform(-100, 100)  # Initial position
    phi = np.random.uniform(0, 2 * np.pi)  # Random initial angle
    theta = np.arctan(np.random.uniform(-0.45, 0.45))  # Random theta angle
    tan_theta = np.tan(theta)  # tan(theta)

    num_points = 10000
    t_range = np.random.uniform(0.5, 4) * np.pi
    t_values = np.linspace(0, t_range, num_points)

    hit_points = []
    first_hit = None
    time_accumulated = start_time
    previous_hit = None

    min_path_length = 2  # Minimum distance between consecutive hits (mm)

    # Loop through trajectory points and calculate hit points
    for i in range(1, len(t_values)):
        t1, t2 = t_values[i - 1], t_values[i]
        x1, y1, z1, R1 = helix(t1, d0, z0, phi, kappa, tan_theta, alpha=spiral_rate)
        x2, y2, z2, R2 = helix(t2, d0, z0, phi, kappa, tan_theta, alpha=spiral_rate)

        if None in [x1, y1, z1, x2, y2, z2]:
            continue

        # Adjust alpha based on the direction of the trajectory (upward or downward)
        if z1 < z2:  # Moving upwards (positive z-direction)
            alpha = abs(spiral_rate)  # Positive for inward spiraling upward
        else:  # Moving downwards (negative z-direction)
            alpha = -abs(spiral_rate)  # Negative for inward spiraling downward

        # Update the helix calculation with the adjusted alpha
        x1, y1, z1, R1 = helix(t1, d0, z0, phi, kappa, tan_theta, alpha=alpha)
        x2, y2, z2, R2 = helix(t2, d0, z0, phi, kappa, tan_theta, alpha=alpha)

        radius = np.sqrt(x2**2 + y2**2)
        if r_min <= radius <= r_max and z_min <= z2 <= z_max:
            phi_hit = np.arctan2(y2, x2)
            phi_hit = (phi_hit + np.pi) % (2 * np.pi)

            for vane_idx in range(num_vanes):
                vane_start = vane_idx * 2 * np.pi / num_vanes
                vane_end = vane_start + delta_phi
                tolerance = delta_phi / 4

                if vane_start - tolerance <= phi_hit < vane_end + tolerance:
                    if previous_hit:
                        x1_prev, y1_prev, z1_prev = previous_hit
                        path_length = np.sqrt(
                            (x2 - x1_prev)**2 + (y2 - y1_prev)**2 + (z2 - z1_prev)**2
                        )
                        if path_length < min_path_length:
                            continue
                        time_increment = path_length / velocity
                        time_accumulated += time_increment
                    else:
                        time_accumulated = start_time

                    timestamp = get_nearest_timestamp(time_accumulated)

                    if should_record_hit():
                        hit_points.append([x2, y2, z2, timestamp, time_accumulated])
                        if first_hit is None:
                            first_hit = [x2, y2, z2, timestamp, time_accumulated]
                            

                    previous_hit = [x2, y2, z2]

    # Add secondary electrons with a probability
    if np.random.rand() < secondary_probability:
        hit_points = add_secondary_electrons_to_random_hit_points(
            hit_points, 
            num_hit_points_range=(0, 5), 
            num_secondary_range=(5, 10), 
            z_variation=10
        )

    # Add noise points around existing hits
    hit_points_with_noise = add_noise_points(hit_points, num_hit_points_with_noise, num_noise_points, noise_radius)

    return np.array(hit_points_with_noise), first_hit


def generate_multiple_tracks_with_pileup(num_intervals, num_secondary, secondary_probability):
    """
    Generate multiple positron tracks with pile-up.

    Args:
        num_intervals (int): Number of time intervals.
        num_secondary (int): Number of secondary electrons to generate.
        secondary_probability (float): Probability of generating secondary electrons.

    Returns:
        tuple: (all_hit_points, first_hits) where all_hit_points is a list of hit points and first_hits is a list of first hit points.
    """
    all_hit_points = []
    first_hits = []
    track_counter = 1  # Continuous track numbering

    for interval in range(num_intervals):
        num_tracks = np.random.randint(30, 51)  # Random number of tracks in each interval
        start_time = interval * 5  # Start time of the interval

        for _ in range(num_tracks):
            print(f"Generating Track {track_counter}...")  # Display track number
            hit_points, first_hit = generate_single_track(
                track_counter, 
                start_time + np.random.uniform(0, 5)
            )

            all_hit_points.append(hit_points)
            first_hits.append(first_hit)
            track_counter += 1  # Increment the track number sequentially

    return all_hit_points, first_hits



# Generate tracks with pile-up
num_intervals = 25
all_hit_points, first_hits = generate_multiple_tracks_with_pileup(num_intervals, num_secondary=0, secondary_probability=secondary_probability)





# After generating tracks, save hit points to Excel
hit_points_data = []
for track_no, hit_points in enumerate(all_hit_points, start=1):  # Ensuring sequential numbering
    for point in hit_points:
        hit_points_data.append([track_no, point[0], point[1], point[2], point[3], point[4]])

df = pd.DataFrame(hit_points_data, columns=['track_no', 'pos_x', 'pos_y', 'pos_z', 'time_ns', 'actual_time_ns'])
df.to_excel(r'C:\Users\niran\Desktop\g-2\track files\hit_points_with_pileup_and_time.xlsx', index=False)




# Plotting 3D hit points
# Create a figure with specific size
fig = plt.figure(figsize=(16, 8))  # Set the figure size (total size of both plots)

# Add the 3D subplot (first subplot)
ax = fig.add_subplot(121, projection='3d')

# Calculate the total number of tracks
total_tracks = sum(1 for hit_points in all_hit_points if hit_points.size > 0)

# Generate a color map based on the total number of tracks
colors = plt.cm.jet(np.linspace(0, 1, total_tracks))

# Counter for color assignment
color_idx = 0

# Plot the hit points for each track in the 3D plot
for hit_points in all_hit_points:
    if hit_points.size > 0:
        ax.scatter(hit_points[:, 0], hit_points[:, 1], hit_points[:, 2], color=colors[color_idx], s=30, label=f'Track {color_idx + 1}')
        color_idx += 1

# Plot the first hit points in red in the 3D plot
for first_hit in first_hits:
    if first_hit is not None:
        ax.scatter(first_hit[0], first_hit[1], first_hit[2], color='red', s=100, marker='*')

# Plot the detector vanes in the 3D plot
for i in range(num_vanes):
    angle_start = i * 2 * np.pi / num_vanes
    angle_end = angle_start + delta_phi
    x_min = r_min * np.cos(angle_start)
    y_min = r_min * np.sin(angle_start)
    x_max = r_max * np.cos(angle_end)
    y_max = r_max * np.sin(angle_end)
    vertices = [[(x_min, y_min, z_min), (x_max, y_max, z_min), (x_max, y_max, z_max), (x_min, y_min, z_max)]]
    poly3d = Poly3DCollection(vertices, facecolors='white', edgecolors='gray', alpha=0.1)
    ax.add_collection3d(poly3d)

# Set axis limits and labels for the 3D plot
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
ax.set_zlim(-200, 200)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('3D Hit Points')

# Create a second subplot for phi vs z (second subplot)
ax_phi_z = fig.add_subplot(122)

# Loop over all hit points and calculate phi = tan^{-1}(y/x)
phi_values = []
z_values = []

color_idx = 0  # Reset color index for the phi vs z plot
for hit_points in all_hit_points:
    if hit_points.size > 0:
        x = hit_points[:, 0]
        y = hit_points[:, 1]
        z = hit_points[:, 2]
        phi = np.arctan2(y, x)  # Calculate phi as arctan(y/x)
        phi_values.extend(phi)
        z_values.extend(z)

        # Plot phi vs z with the same color as in the 3D plot
        ax_phi_z.scatter(phi, z, color=colors[color_idx], s=20)  # Use the same color for each track
        color_idx += 1

# Set axis labels and title for phi vs z plot
ax_phi_z.set_xlabel('Phi (radians)')
ax_phi_z.set_ylabel('Z (mm)')
ax_phi_z.set_title('Phi vs Z')
ax_phi_z.set_ylim(-200, 200)
ax_phi_z.set_xlim(-3.5 , 3.5)  # Set the y-axis (Z) range from -200 to 200
ax_phi_z.grid(True)

# Adjust layout for better spacing between the subplots
plt.tight_layout()

# Show the plot
plt.show()

