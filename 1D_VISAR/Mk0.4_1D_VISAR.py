import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

def simulate_visar_streak_camera_realistic(
    nx=500,
    nt=500,
    x_range=(-460, -300),    # Position in microns [µm]
    t_range=(12, 22),        # Time in nanoseconds [ns]
    wavelength=0.532,        # Laser wavelength in microns (532 nm = 0.532 µm)
    n_etalon=1.5,            # Refractive index inside the etalon
    d_etalon=1.0,            # Etalon thickness in microns
    R=0.8,                   # Reflectivity at each etalon surface (0 < R < 1)
    alpha_deg=0.0,           # Tilt angle (in degrees) for the spatial fringe pattern
    K=5.0,                   # Spatial frequency in [1/µm]
    omega=0.0,               # Extra time-dependent phase frequency [rad/ns] (optional)
    phi_etalon=0.0,          # Additional constant phase shift from the etalon
    velocity_profile=None,   # A callable v(t) or array specifying velocity [µm/ns] over time [ns]
    add_noise=False,         # If True, add random noise to mimic real data
    noise_level=0.05,        # Noise amplitude relative to max signal
    fringe_scale=1.0,        # Scales all phases to control the number of fringes
    show_plot=True
):
    """
    Simulates a VISAR streak camera output with realistic units and an optional phase scaling factor.
    
    Parameters:
      nx, nt           : Number of points in x (µm) and time (ns).
      x_range          : Spatial extent (start, end) in microns.
      t_range          : Time extent (start, end) in ns.
      wavelength       : Laser wavelength in microns.
      n_etalon         : Refractive index inside the etalon.
      d_etalon         : Etalon thickness in microns.
      R                : Reflectivity at each etalon surface.
      alpha_deg        : Tilt angle in degrees (fringe orientation).
      K                : Spatial frequency in [1/µm].
      omega            : Extra time-dependent phase frequency [rad/ns].
      phi_etalon       : Constant phase offset from the etalon.
      velocity_profile : Custom velocity profile as a function or array (in µm/ns).
      add_noise        : If True, adds random noise to the final image.
      noise_level      : Amplitude of noise relative to the maximum signal.
      fringe_scale     : Multiplies all phases, allowing you to control the visible number of fringes.
      show_plot        : If True, displays the resulting streak camera image.
      
    Returns:
      I_streak         : 2D numpy array (nt x nx) of the simulated streak camera intensity.
      x                : 1D array of spatial coordinates [µm].
      t                : 1D array of time coordinates [ns].
    """
    # 1) Create Spatial and Temporal Grids in Real Units
    x = np.linspace(x_range[0], x_range[1], nx)   # microns
    t = np.linspace(t_range[0], t_range[1], nt)     # nanoseconds

    # 2) Convert Tilt Angle -> Radians; Decompose Spatial Frequency
    alpha_rad = np.deg2rad(alpha_deg)
    kx = K * np.cos(alpha_rad)  # [1/µm]

    # 3) Compute Etalon Transmission (Fabry–Perot, Normal Incidence)
    delta_single = (2 * np.pi / wavelength) * n_etalon * d_etalon
    T_etalon = ((1 - R)**2) / (1 - 2 * R * np.cos(2 * delta_single) + R**2)
    A_etalon = np.sqrt(T_etalon)

    # 4) Determine Velocity Profile (in µm/ns)
    t0, t1 = t_range
    if velocity_profile is None:
        # Default: piecewise profile (for demonstration)
        def default_v(t_ns):
            v_vals = np.zeros_like(t_ns)
            for i, ti in enumerate(t_ns):
                if 15 <= ti < 18:
                    v_vals[i] = 2.0 * (ti - 15) / (18 - 15)
                elif 18 <= ti < 20:
                    v_vals[i] = 2.0
                elif 20 <= ti <= 22:
                    v_vals[i] = 2.0 - (ti - 20) * (1.0 / 2.0)
            return v_vals
        v_profile = default_v(t)
    else:
        if callable(velocity_profile):
            v_profile = velocity_profile(t)
        else:
            v_profile = np.array(velocity_profile)
        if len(v_profile) != nt:
            raise ValueError("velocity_profile must match nt in length.")

    # 5) Doppler Phase via Integrated Velocity (4π/λ * ∫v dt)
    integrated_velocity = cumulative_trapezoid(v_profile, t, initial=0)
    phi_doppler = (4 * np.pi / wavelength) * integrated_velocity  # shape (nt,)
    
    # 6) Create 2D Grids for x and t
    X = x[None, :]   # shape (1, nx)
    T = t[:, None]   # shape (nt, 1)
    
    # 7) Define Interferometer Fields
    phi_doppler_2D = phi_doppler[:, None]
    phi_total = kx * X + omega * T + phi_etalon + phi_doppler_2D
    
    # Scale the phase to control the number of visible fringes
    phi_total_scaled = fringe_scale * phi_total
    
    E1 = 1.0  # Reference arm
    E2 = A_etalon * np.exp(1j * phi_total_scaled)
    
    E_total = E1 + E2
    I_streak = np.abs(E_total)**2

    # 8) Optionally Add Noise
    if add_noise:
        max_signal = I_streak.max()
        noise = noise_level * max_signal * np.random.randn(*I_streak.shape)
        I_streak += noise
        I_streak = np.clip(I_streak, 0, None)

    # 9) Plot the Result if Desired
    if show_plot:
        plt.figure(figsize=(7, 5))
        plt.imshow(I_streak,
                   extent=[x_range[0], x_range[1], t_range[0], t_range[1]],
                   origin='lower', aspect='auto', cmap='gray')
        plt.colorbar(label='Intensity')
        plt.xlabel('X [µm]')
        plt.ylabel('Time [ns]')
        plt.title('Simulated VISAR Streak Camera Output (Fringe-Scaled)')
        plt.show()

    return I_streak, x, t

# --------------------------------------------------------------------
# Define a test quadratic velocity profile (v(t) ∝ (t-t0)²)
def quadratic_velocity_profile(t):
    t0, t1 = t[0], t[-1]
    # Scale so that the maximum velocity at t1 is 2 µm/ns
    return np.sin(t/2) #2.0 * ((t - t0) / (t1 - t0))**2

# --------------------------------------------------------------------
# Example usage: simulate using the quadratic velocity profile
if __name__ == "__main__":
    I, xvals, tvals = simulate_visar_streak_camera_realistic(
        nx=600,
        nt=600,
        x_range=(-460, -300),     
        t_range=(12, 100),        
        wavelength=0.532,        
        n_etalon=1.5,
        d_etalon=1.0,
        R=0.8,
        alpha_deg=0.0,
        K=5.0,
        omega=0.0,
        phi_etalon=0.0,
        velocity_profile=quadratic_velocity_profile,  # Use the quadratic velocity profile
        add_noise=True,
        noise_level=0.05,
        fringe_scale=0.1,  
        show_plot=True
    )
