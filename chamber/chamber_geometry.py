import numpy as np
import matplotlib.pyplot as plt
import sys



from nozzle_solver import rao 

theta_default = np.pi/4
force_coeffcient_default = 1.4
diameter_exit_default = 4 / 39.37
l_star_default = 1.27
chamber_diameter_default = 3.4 / 39.37

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

def chamber_volume_calc(area_throat, l_star = l_star_default):
    """
    Parameters:
    - l_star: The characteristic length of the chamber. (Get from config)
    - area_throat: The area of the throat of the chamber.
    Calculate the volume of the chamber.
    """


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

def area_chamber_calc(diameter_inner=chamber_diameter_default):
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


def contraction_length_calc(area_chamber, entrance_arc_start_y, theta=theta_default):
  
    R_start = np.sqrt(area_chamber / np.pi)
    R_end = entrance_arc_start_y
    L_cone = (R_start - R_end) * np.tan(np.pi/2 - theta)
    return L_cone

def contraction_length_actual_calc(length, entrance_arc_offset):
    """
    accounts for horizontal difference bc of the arc region of nozzle
    """
    return length - entrance_arc_offset


def generate_nozzle(area_throat, area_exit, steps=200):
    return rao(area_throat, area_exit, method="top", do_plot=False, steps=steps)


def generate_chamber_contour(area_chamber, area_throat, length_cylindrical, 
                             entrance_arc_start_x, entrance_arc_start_y, 
                             theta=theta_default, steps=200):
    """
    Generate chamber contour points going backwards from the nozzle entrance arc start.
    The chamber consists of:
    1. A cylindrical section
    2. A 45deg line connecting to the entrance arc start point
    
    Parameters:
    -----------
    area_chamber : float
        Chamber cross-sectional area [m²]
    area_throat : float
        Throat area [m²]
    length_cylindrical : float
        Length of cylindrical section [m]
    entrance_arc_start_x : float
        x coordinate where entrance arc starts (negative value)
    entrance_arc_start_y : float
        y coordinate where entrance arc starts
    theta : float
        Angle of the 45deg line [rad] (should be pi/4)
    steps : int
        Number of points for each section
    
    Returns:
    --------
    pts_chamber : ndarray
        Array of (x, y) points for chamber contour (going backwards, so x decreases)
    """
    R_c = np.sqrt(area_chamber / np.pi)
    
    # The 45deg line connects from the entrance arc start to the cylindrical section
    # At the entrance arc start: (entrance_arc_start_x, entrance_arc_start_y)
    # At the cylindrical section start: (x_cyl_start, R_c)
    # 
    # The 45deg line has slope = -1 (going backwards and up)
    # Equation: y = y_start + slope * (x - x_start)
    # With slope = -1: y = y_start - (x - x_start) = y_start + x_start - x
    #
    # At the cylindrical section, y = R_c, so:
    # R_c = y_start + x_start - x_cyl_start
    # x_cyl_start = y_start + x_start - R_c
    
    x_cyl_start = entrance_arc_start_x + entrance_arc_start_y - R_c
    
    # Generate 45deg line (going backwards from entrance arc start)
    # x decreases from entrance_arc_start_x to x_cyl_start
    x_line = np.linspace(entrance_arc_start_x, x_cyl_start, steps)
    # y = y_start + x_start - x (45deg line going backwards and up, slope = -1)
    y_line = entrance_arc_start_y + entrance_arc_start_x - x_line
    pts_line = np.column_stack((x_line, y_line))
    
    # Generate cylindrical section (going further backwards)
    x_cyl_end = x_cyl_start - length_cylindrical
    x_cyl = np.linspace(x_cyl_start, x_cyl_end, steps)
    y_cyl = np.full(steps, R_c)
    pts_cyl = np.column_stack((x_cyl, y_cyl))
    
    # Combine: cylindrical, then 45deg line (going backwards, so reverse order)
    # Reverse order so we go from injector face (most negative x) to entrance arc
    pts_cyl_reversed = pts_cyl[::-1]
    pts_line_reversed = pts_line[::-1]
    pts_chamber = np.vstack([pts_cyl_reversed, pts_line_reversed[1:]])
    
    # Return the chamber points and the index where the 45deg line starts
    # The 45deg line starts after the cylindrical section
    cyl_end_idx = len(pts_cyl_reversed)
    line_start_idx = cyl_end_idx
    
    return pts_chamber, cyl_end_idx, line_start_idx


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
        - pts_chamber: Chamber contour points (Nx2 array)
        - pts_nozzle: Nozzle contour points (Nx2 array, x=0 at throat)
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
    
    # Get nozzle contour (without plotting) - keep it unchanged, x=0 at throat
    nozzle_pts, x_first_nozzle = rao(area_throat, area_exit, method="top", do_plot=False, steps=steps)
    
    # Get the entrance arc start point (where chamber should connect)
    entrance_arc_start_x, entrance_arc_start_y, tangent_slope = get_nozzle_entrance_arc_start(area_throat)
    
    # Generate chamber contour going backwards from entrance arc start
    pts_chamber, cyl_end_idx, line_start_idx = generate_chamber_contour(
        area_chamber, area_throat, length_cylindrical,
        entrance_arc_start_x, entrance_arc_start_y, theta, steps
    )
    
    # Nozzle is unchanged (x=0 at throat)
    # Combine full contour: chamber (going backwards) + nozzle (starting at entrance arc)
    # The last point of chamber should connect to the first point of nozzle
    pts_full = np.vstack([pts_chamber, nozzle_pts[1:]])  # Skip first nozzle point to avoid duplicate
    
    # Plot if requested
    if do_plot:
        plt.figure(figsize=(12, 6))
        
        if color_segments:
            # Chamber cylindrical section
            plt.plot(pts_chamber[:cyl_end_idx, 0], pts_chamber[:cyl_end_idx, 1], 
                    label='Chamber (cylindrical)', color='purple', linewidth=2)
            # Chamber 45deg line section
            plt.plot(pts_chamber[line_start_idx:, 0], pts_chamber[line_start_idx:, 1], 
                    label='Chamber (45° line)', color='orange', linewidth=2)
            # Nozzle (unchanged, x=0 at throat)
            plt.plot(nozzle_pts[:, 0], nozzle_pts[:, 1], 
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
        'pts_chamber': pts_chamber,
        'pts_nozzle': nozzle_pts,  # Nozzle unchanged, x=0 at throat
        'pts_full': pts_full
    }


results = chamber_solver(
    pc_design=2.068e6,  # 1 MPa
    thrust_design=6000,  # 1000 N
    do_plot=True,
    color_segments=True
)

# Access full contour points
print(results)
pts_full = results['pts_full']  # Nx2 array of (x, y) points