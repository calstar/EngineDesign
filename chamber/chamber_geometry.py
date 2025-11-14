import numpy as np
import matplotlib.pyplot as plt
import sys



from nozzle_solver import rao 

theta_default = np.pi/4
force_coeffcient_default = 1.4
diameter_exit_default = 4 / 39.37

def area_exit_calc(diameter_exit=diameter_exit_default):
    """
    Calculate the area of the exit of the chamber.
    Parameters:
    - diameter_exit: The diameter of the exit of the chamber.
    Calculate the area of the exit of the chamber.
    """
    return np.pi * (diameter_exit / 2) ** 2

def expansion_ratio_calc(area_exit, area_throat):
    """
    Calculate the expansion ratio of the chamber.
    Parameters:
    - area_exit: The area of the exit of the chamber.
    - area_throat: The area of the throat of the chamber.
    Calculate the expansion ratio of the chamber.
    """
    return area_exit / area_throat

def area_throat_calc(pc_design, thrust_design, force_coeffcient=force_coeffcient_default):
    """
    Calculate the area of the throat of the chamber.
    Parameters:
    - pc_design: The design chamber pressure.
    - thrust_design: The design thrust.
    - force_coeffcient: The force coeffcient.
    Calculate the area of the throat of the chamber.
    """
    return thrust_design / (pc_design * force_coeffcient)

def chamber_volume_calc(area_throat, l_star = None):
    """
    Parameters:
    - l_star: The characteristic length of the chamber. (Get from config)
    - area_throat: The area of the throat of the chamber.
    Calculate the volume of the chamber.
    """
    if l_star is None:
        l_star = 1.27

    return l_star * area_throat




def contraction_ratio_calc(area_chamber, area_throat):
    """
    Calculate the contraction ratio of the chamber.
    Parameters:
    - area_chamber: The area of the chamber.
    - area_throat: The area of the throat of the chamber.
    Calculate the contraction ratio of the chamber.
    """
    return area_chamber / area_throat

def area_chamber_calc(diameter_inner=3.4 / 39.37):
    """
    Calculate the area of the chamber.
    Parameters:
    - diameter_inner: The inner diameter of the chamber. = 3.4"
    Calculate the area of the chamber.
    """
    return np.pi * (diameter_inner / 2) ** 2

def chamber_diameter_calc(area_chamber):
    """
    Calculate the diameter of the chamber.
    Parameters:
    - area_chamber: The area of the chamber.
    Calculate the diameter of the chamber.
    """
    return np.sqrt(4 * area_chamber / np.pi)

def chamber_length_calc(chamber_volume, area_throat, contraction_ratio, theta = theta_default):
    """
    Calculate the length of the chamber.
    Parameters:
    - chamber_volume: The volume of the chamber.
    - area_throat: The area of the throat of the chamber.
    - contraction_ratio: The contraction ratio of the chamber.
    - theta: The angle of the chamber. = 45 degrees from -135deg nozzle entrance
    Calculate the length of the chamber.
    """
    t1 = (chamber_volume / area_throat)
    t2 = (1/3)*np.sqrt(area_throat / np.pi) * (1/np.tan(theta)) * (contraction_ratio**(1/3) - 1)
    t3 = t1 - t2
    t4 = t3 / contraction_ratio
    return t4


def contraction_length_calc(area_chamber, area_throat, theta=theta_default):
    """
    Calculate the horizontal length of the conical contraction section
    from the end of the cylindrical chamber to the throat.
    
    Parameters:
    -----------
    area_chamber : float
        Chamber area [m²]
    area_throat : float
        Throat area [m²]
    theta : float
        Half-angle of conical contraction [rad]
    
    Returns:
    --------
    L_cone : float
        Horizontal length of contraction section [m]
    """
    R_c = np.sqrt(area_chamber / np.pi)
    R_t = np.sqrt(area_throat / np.pi)
    L_cone = (R_c - R_t) / np.tan(theta)
    return L_cone

def contraction_length_actual_calc(length, entrance_arc_offset):
    """
    accounts for horizontal difference bc of the arc region of nozzle
    """
    return length - entrance_arc_offset


def generate_chamber_contour(area_chamber, area_throat, length_cylindrical, contraction_length, theta=theta_default, steps=200):
    """
    Generate chamber contour points (cylindrical + conical contraction sections).
    
    Parameters:
    -----------
    area_chamber : float
        Chamber cross-sectional area [m²]
    area_throat : float
        Throat area [m²]
    length_cylindrical : float
        Length of cylindrical section [m]
    contraction_length : float
        Horizontal length of conical contraction section [m]
    theta : float
        Half-angle of conical contraction [rad]
    steps : int
        Number of points for each section
    
    Returns:
    --------
    pts_chamber : ndarray
        Array of (x, y) points for chamber contour
    """
    R_c = np.sqrt(area_chamber / np.pi)
    R_t = np.sqrt(area_throat / np.pi)
    
    # Cylindrical section (from injector face)
    x_cyl = np.linspace(0, length_cylindrical, steps)
    y_cyl = np.full(steps, R_c)
    pts_cyl = np.column_stack((x_cyl, y_cyl))
    
    # Conical contraction section
    x_cone_start = length_cylindrical
    x_cone = np.linspace(x_cone_start, x_cone_start + contraction_length, steps)
    # Linear interpolation from R_c to R_t
    y_cone = R_c - (R_c - R_t) * (x_cone - x_cone_start) / contraction_length
    pts_cone = np.column_stack((x_cone, y_cone))
    
    # Combine (skip first point of cone to avoid duplicate)
    pts_chamber = np.vstack([pts_cyl, pts_cone[1:]])
    
    return pts_chamber


def chamber_solver(pc_design, thrust_design, force_coeffcient=force_coeffcient_default, 
                   do_plot=False, color_segments=False, theta=theta_default, steps=200):
    """
    Solve for chamber geometry and generate full contour (chamber + nozzle).
    
    Parameters:
    -----------
    pc_design : float
        Design chamber pressure [Pa]
    thrust_design : float
        Design thrust [N]
    force_coeffcient : float
        Force coefficient (default: 1.4)
    do_plot : bool
        Whether to plot the contour
    color_segments : bool
        Whether to color-code segments in plot
    theta : float
        Half-angle of conical contraction [rad]
    steps : int
        Number of points per section
    
    Returns:
    --------
    results : dict
        Dictionary containing:
        - area_throat: Throat area [m²]
        - volume_chamber: Chamber volume [m³]
        - area_chamber: Chamber area [m²]
        - contraction_ratio: Contraction ratio
        - length: Cylindrical section length [m]
        - contraction_length: Conical contraction length [m]
        - contraction_length_actual: Actual contraction length accounting for arc [m]
        - pts_chamber: Chamber contour points (Nx2 array)
        - pts_nozzle: Nozzle contour points (Nx2 array)
        - pts_full: Full contour points (chamber + nozzle) (Nx2 array)
    """
    # Calculate geometry
    area_throat = area_throat_calc(pc_design, thrust_design, force_coeffcient)
    area_exit = area_exit_calc(diameter_exit_default)
    expansion_ratio = expansion_ratio_calc(area_exit, area_throat)
    volume_chamber = chamber_volume_calc(area_throat)
    area_chamber = area_chamber_calc()
    contraction_ratio = contraction_ratio_calc(area_chamber, area_throat)
    length_cylindrical = chamber_length_calc(volume_chamber, area_throat, contraction_ratio, theta)
    contraction_length = contraction_length_calc(area_chamber, area_throat, theta)
    
    # Get nozzle contour (without plotting)
    nozzle_pts, x_first_nozzle = rao(area_throat, area_exit, method="top", do_plot=False, steps=steps)
    arc_offset = x_first_nozzle
    contraction_length_actual = contraction_length_actual_calc(contraction_length, arc_offset)
    
    # Generate chamber contour
    pts_chamber = generate_chamber_contour(
        area_chamber, area_throat, length_cylindrical, 
        contraction_length_actual, theta, steps
    )
    
    # Find split point between cylindrical and conical sections
    # The cylindrical section ends where x = length_cylindrical
    split_idx = np.searchsorted(pts_chamber[:, 0], length_cylindrical)
    if split_idx >= len(pts_chamber):
        split_idx = len(pts_chamber) // 2
    
    # Shift nozzle points to connect with chamber
    # The nozzle starts at x=0, we need to shift it to start where chamber ends
    x_chamber_end = pts_chamber[-1, 0]
    nozzle_pts_shifted = nozzle_pts.copy()
    nozzle_pts_shifted[:, 0] += x_chamber_end
    
    # Combine full contour
    pts_full = np.vstack([pts_chamber, nozzle_pts_shifted[1:]])  # Skip first nozzle point to avoid duplicate
    
    # Plot if requested
    if do_plot:
        plt.figure(figsize=(12, 6))
        
        if color_segments:
            # Chamber cylindrical section
            plt.plot(pts_chamber[:split_idx, 0], pts_chamber[:split_idx, 1], 
                    label='Chamber (cylindrical)', color='purple', linewidth=2)
            # Chamber conical section
            plt.plot(pts_chamber[split_idx:, 0], pts_chamber[split_idx:, 1], 
                    label='Chamber (conical)', color='orange', linewidth=2)
            # Nozzle
            plt.plot(nozzle_pts_shifted[:, 0], nozzle_pts_shifted[:, 1], 
                    label='Nozzle', color='blue', linewidth=2)
            plt.legend()
        else:
            plt.plot(pts_full[:, 0], pts_full[:, 1], 'k-', linewidth=2)
        
        plt.gca().set_aspect('equal', 'box')
        plt.xlabel("Axial distance x [m]")
        plt.ylabel("Radius y [m]")
        plt.grid(True, alpha=0.3)
        plt.title("Full Chamber + Nozzle Contour")
        
        try:
            plt.savefig('chamber_full_contour.png', dpi=150, bbox_inches='tight')
            print("Plot saved to chamber_full_contour.png")
        except Exception:
            pass
        finally:
            plt.close()
    
    return {
        'area_throat': area_throat,
        'volume_chamber': volume_chamber,
        'area_chamber': area_chamber,
        'contraction_ratio': contraction_ratio,
        'length': length_cylindrical,
        'contraction_length': contraction_length,
        'contraction_length_actual': contraction_length_actual,
        'pts_chamber': pts_chamber,
        'pts_nozzle': nozzle_pts_shifted,
        'pts_full': pts_full
    }


results = chamber_solver(
    pc_design=2.068e6,  # 1 MPa
    thrust_design=6000,  # 1000 N
    do_plot=True,
    color_segments=True
)

# Access full contour points
pts_full = results['pts_full']  # Nx2 array of (x, y) points