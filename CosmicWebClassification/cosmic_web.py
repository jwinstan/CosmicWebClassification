import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import njit, prange



@njit #@njit(parallel=True)
def ngp_numba(positions, velocities, box_size,
              vel_x, vel_y, vel_z, count):
    N = positions.shape[0]
    grid_size = vel_x.shape[0]
    for p in range(N): #prange(N)?????
        
        pos_frac_x = positions[p, 0] / box_size
        pos_frac_y = positions[p, 1] / box_size
        pos_frac_z = positions[p, 2] / box_size

        ix = int(np.floor(pos_frac_x * grid_size)) % grid_size
        iy = int(np.floor(pos_frac_y * grid_size)) % grid_size
        iz = int(np.floor(pos_frac_z * grid_size)) % grid_size

        vel_x[ix, iy, iz] += velocities[p, 0]
        vel_y[ix, iy, iz] += velocities[p, 1]
        vel_z[ix, iy, iz] += velocities[p, 2]
        count[ix, iy, iz] += 1.0
    
@njit
def cic_numba(positions, velocities, box_size,
              vel_x, vel_y, vel_z, count):
    N = positions.shape[0]
    grid_size = vel_x.shape[0]
    for p in range(N): #prange???????????????
        scaled_pos = positions[p] / box_size * grid_size
        base = np.floor(scaled_pos).astype(np.int64)
        frac = scaled_pos - base

        for dx in (0, 1):
            wx = (1.0 - frac[0]) if dx == 0 else frac[0]
            ix = (base[0] + dx) % grid_size
            for dy in (0, 1):
                wy = (1.0 - frac[1]) if dy == 0 else frac[1]
                iy = (base[1] + dy) % grid_size
                for dz in (0, 1):
                    wz = (1.0 - frac[2]) if dz == 0 else frac[2]
                    iz = (base[2] + dz) % grid_size

                    w = wx * wy * wz
                    vel_x[ix, iy, iz] += velocities[p, 0] * w
                    vel_y[ix, iy, iz] += velocities[p, 1] * w
                    vel_z[ix, iy, iz] += velocities[p, 2] * w
                    count[ix, iy, iz] += w

@njit
def tsc_kernel(x):
    ax = np.abs(x)
    w = 0.0
    if ax < 0.5:
        w = 0.75 - ax**2
    elif ax < 1.5:
        w = 0.5 * (1.5 - ax)**2
    return w

@njit
def tsc_numba(positions, velocities, box_size,
              vel_x, vel_y, vel_z, count):
    N = positions.shape[0]
    grid_size = vel_x.shape[0]

    # shifts to cover 3x3x3 neighbors
    shifts = np.array([-1, 0, 1], dtype=np.int64)

    for p in range(N): #prange????
        scaled_pos = positions[p] / box_size * grid_size
        base = np.floor(scaled_pos).astype(np.int64)

        for sx in shifts:
            ix = (base[0] + sx) % grid_size
            dx = scaled_pos[0] - (base[0] + sx + 0.5)
            wx = tsc_kernel(dx)

            for sy in shifts:
                iy = (base[1] + sy) % grid_size
                dy = scaled_pos[1] - (base[1] + sy + 0.5)
                wy = tsc_kernel(dy)

                for sz in shifts:
                    iz = (base[2] + sz) % grid_size
                    dz = scaled_pos[2] - (base[2] + sz + 0.5)
                    wz = tsc_kernel(dz)

                    w = wx * wy * wz
                    vel_x[ix, iy, iz] += velocities[p, 0] * w
                    vel_y[ix, iy, iz] += velocities[p, 1] * w
                    vel_z[ix, iy, iz] += velocities[p, 2] * w
                    count[ix, iy, iz] += w
    return vel_x, vel_y, vel_z, count

@njit
def ngp_mass_numba(positions, masses, box_size, mass_grid):
    N = positions.shape[0]
    grid_size = mass_grid.shape[0]
    for p in range(N):
        pos_frac_x = positions[p, 0] / box_size
        pos_frac_y = positions[p, 1] / box_size
        pos_frac_z = positions[p, 2] / box_size

        ix = int(np.floor(pos_frac_x * grid_size)) % grid_size
        iy = int(np.floor(pos_frac_y * grid_size)) % grid_size
        iz = int(np.floor(pos_frac_z * grid_size)) % grid_size

        mass_grid[ix, iy, iz] += masses[p]

@njit
def cic_mass_numba(positions, masses, box_size, mass_grid):
    N = positions.shape[0]
    grid_size = mass_grid.shape[0]

    for p in range(N):
        scaled_pos = positions[p] / box_size * grid_size
        base = np.floor(scaled_pos).astype(np.int64)
        frac = scaled_pos - base

        for dx in (0, 1):
            wx = (1.0 - frac[0]) if dx == 0 else frac[0]
            ix = (base[0] + dx) % grid_size
            for dy in (0, 1):
                wy = (1.0 - frac[1]) if dy == 0 else frac[1]
                iy = (base[1] + dy) % grid_size
                for dz in (0, 1):
                    wz = (1.0 - frac[2]) if dz == 0 else frac[2]
                    iz = (base[2] + dz) % grid_size

                    w = wx * wy * wz
                    mass_grid[ix, iy, iz] += masses[p] * w

@njit
def tsc_mass_numba(positions, masses, box_size, mass_grid):
    N = positions.shape[0]
    grid_size = mass_grid.shape[0]
    shifts = np.array([-1, 0, 1], dtype=np.int64)

    for p in range(N):
        scaled_pos = positions[p] / box_size * grid_size
        base = np.floor(scaled_pos).astype(np.int64)

        for sx in shifts:
            ix = (base[0] + sx) % grid_size
            dx = scaled_pos[0] - (base[0] + sx + 0.5)
            wx = tsc_kernel(dx)

            for sy in shifts:
                iy = (base[1] + sy) % grid_size
                dy = scaled_pos[1] - (base[1] + sy + 0.5)
                wy = tsc_kernel(dy)

                for sz in shifts:
                    iz = (base[2] + sz) % grid_size
                    dz = scaled_pos[2] - (base[2] + sz + 0.5)
                    wz = tsc_kernel(dz)

                    w = wx * wy * wz
                    mass_grid[ix, iy, iz] += masses[p] * w

def build_velocity_grid_numba(positions: np.ndarray,
                        velocities: np.ndarray,
                        box_size: float,
                        grid_size: int = 100,
                        method: str = "cic",
                        vel_x: np.ndarray = None,
                        vel_y: np.ndarray = None,
                        vel_z: np.ndarray = None,
                        count: np.ndarray = None):
    """
    Bin particle velocities into a 3D grid with different assignment schemes.

    Parameters
    ----------
    positions : (N, 3) array
        Particle positions, assumed inside [0, box_size).
    velocities : (N, 3) array
        Particle velocities.
    box_size : float
        Size of the simulation box (same units as positions).
    grid_size : int
        Number of cells per dimension (default 100).
    method : str, optional
        Interpolation scheme:
          - "ngp": Nearest Grid Point 
          - "cic": Cloud In Cell (default)
          - "tsc": Triangular Shaped Cloud
    vel_x, vel_y, vel_z : optional (grid_size, grid_size, grid_size) arrays
        Arrays to accumulate velocity sums. If None, new arrays are created.
    count : optional (grid_size, grid_size, grid_size) array
        Array to accumulate counts or weights. If None, a new array is created.
    average : optional, bool
        This will return the weighted averages instead of the sums.

    Returns
    -------
    velocity_x, velocity_y, velocity_z : (grid_size, grid_size, grid_size) arrays
        Smoothed weighted velocity components in each cell.
    count : (grid_size, grid_size, grid_size) array
        Number of particles (or total weight) assigned to each cell.
    """

    provided = [vel_x, vel_y, vel_z, count]
    if any(x is None for x in provided) and not all(x is None for x in provided):
        raise ValueError("If providing velocity grids, you must provide all of vel_x, vel_y, vel_z, and count.")

    # Create arrays if none provided _> Used if not called in class.
    if all(x is None for x in provided):
        vel_x = np.zeros((grid_size, grid_size, grid_size), dtype=float)
        vel_y = np.zeros_like(vel_x)
        vel_z = np.zeros_like(vel_x)
        count = np.zeros_like(vel_x)

    if method == "ngp":
        ngp_numba(positions, velocities, box_size,
                  vel_x, vel_y, vel_z, count)
    elif method == "cic":
        cic_numba(positions, velocities, box_size,
                  vel_x, vel_y, vel_z, count)
    elif method == "tsc":
        tsc_numba(positions, velocities, box_size,
                  vel_x, vel_y, vel_z, count)
    else:
        raise ValueError(f"Unknown method '{method}', please use 'ngp', 'cic', or 'tsc'.")

    return vel_x, vel_y, vel_z, count

def build_mass_grid_numba(positions: np.ndarray,
                    masses: np.ndarray,
                    box_size: float,
                    grid_size: int = 100,
                    method: str = "cic",
                    mass_grid:np.ndarray = None):
    """
    Bin particle masses into a 3D grid with different assignment schemes.

    Parameters
    ----------
    positions : (N, 3) array
        Particle positions, assumed inside [0, box_size).
    masses : (N,) array
        Particle masses.
    box_size : float
        Size of the simulation box (same units as positions).
    grid_size : int
        Number of cells per dimension (default 100).
    method : str, optional
        Interpolation scheme:
          - "ngp": Nearest Grid Point 
          - "cic": Cloud In Cell (default)
          - "tsc": Triangular Shaped Cloud
    mass_grid : np.ndarray, optional
        If no grid is passed in, it will construct a new one.

    Returns
    -------
    mass_grid : (grid_size, grid_size, grid_size) array
        Mass in each cell.
    """

    if mass_grid is None:
        mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    if method == "ngp":
        ngp_mass_numba(positions, masses, box_size, mass_grid)
    elif method == "cic":
        cic_mass_numba(positions, masses, box_size, mass_grid)
    elif method == "tsc":
        tsc_mass_numba(positions, masses, box_size, mass_grid)
    else:
        raise ValueError(f"Unknown method '{method}'")

    return mass_grid




def build_velocity_grid(positions: np.ndarray,
                        velocities: np.ndarray,
                        box_size: float,
                        grid_size: int = 100,
                        method: str = "ngp",
                        vel_x: np.ndarray = None,
                        vel_y: np.ndarray = None,
                        vel_z: np.ndarray = None,
                        count: np.ndarray = None):
    """
    Bin particle velocities into a 3D grid with different assignment schemes.

    Parameters
    ----------
    positions : (N, 3) array
        Particle positions, assumed inside [0, box_size).
    velocities : (N, 3) array
        Particle velocities.
    box_size : float
        Size of the simulation box (same units as positions).
    grid_size : int
        Number of cells per dimension (default 100).
    method : str, optional
        Interpolation scheme:
          - "ngp": Nearest Grid Point (default)
          - "tsc": Triangular Shaped Cloud
          - "cic": Cloud In Cell
    vel_x, vel_y, vel_z : optional (grid_size, grid_size, grid_size) arrays
        Arrays to accumulate velocity sums. If None, new arrays are created.
    count : optional (grid_size, grid_size, grid_size) array
        Array to accumulate counts or weights. If None, a new array is created.
    average : optional, bool
        This will return the weighted averages instead of the sums.

    Returns
    -------
    velocity_x, velocity_y, velocity_z : (grid_size, grid_size, grid_size) arrays
        Smoothed weighted velocity components in each cell.
    count : (grid_size, grid_size, grid_size) array
        Number of particles (or total weight) assigned to each cell.
    """

    provided = [vel_x, vel_y, vel_z, count]
    if any(x is None for x in provided) and not all(x is None for x in provided):
        raise ValueError("If providing velocity grids, you must provide all of vel_x, vel_y, vel_z, and count.")

    # Create arrays if none provided
    if all(x is None for x in provided):
        vel_x = np.zeros((grid_size, grid_size, grid_size), dtype=float)
        vel_y = np.zeros_like(vel_x)
        vel_z = np.zeros_like(vel_x)
        count = np.zeros_like(vel_x)

    pos_frac = positions / box_size

    if method == "ngp":
        # Nearest Grid Point
        indices = np.floor(pos_frac * grid_size).astype(int)
        indices = np.mod(indices, grid_size)  # periodic wrap

        np.add.at(vel_x, tuple(indices.T), velocities[:, 0])
        np.add.at(vel_y, tuple(indices.T), velocities[:, 1])
        np.add.at(vel_z, tuple(indices.T), velocities[:, 2])
        np.add.at(count, tuple(indices.T), 1)

    elif method == "tsc":
        def tsc_kernel(x):
            ax = np.abs(x)
            w = np.zeros_like(ax)
            m1 = ax < 0.5
            w[m1] = 0.75 - ax[m1]**2
            m2 = (ax >= 0.5) & (ax < 1.5)
            w[m2] = 0.5 * (1.5 - ax[m2])**2
            return w
        scaled_pos = pos_frac * grid_size
        base = np.floor(scaled_pos).astype(int)
        shifts = np.array([[-1,-1,-1], [-1,-1,0], [-1,-1,1],
                           [-1, 0,-1], [-1, 0,0], [-1, 0,1],
                           [-1, 1,-1], [-1, 1,0], [-1, 1,1],
                           [ 0,-1,-1], [ 0,-1,0], [ 0,-1,1],
                           [ 0, 0,-1], [ 0, 0,0], [ 0, 0,1],
                           [ 0, 1,-1], [ 0, 1,0], [ 0, 1,1],
                           [ 1,-1,-1], [ 1,-1,0], [ 1,-1,1],
                           [ 1, 0,-1], [ 1, 0,0], [ 1, 0,1],
                           [ 1, 1,-1], [ 1, 1,0], [ 1, 1,1]], dtype=int)

        for sx, sy, sz in shifts:
            neigh = (base + (sx, sy, sz)) 

            d = scaled_pos - (neigh + 0.5)
            d = (d + grid_size/2) % grid_size - grid_size/2
            #d = np.where(d >  grid_size/2, d - grid_size, d)
            #d = np.where(d < -grid_size/2, d + grid_size, d)

            w = tsc_kernel(d[:, 0]) * tsc_kernel(d[:, 1]) * tsc_kernel(d[:, 2])

            ix = neigh[:,0] % grid_size
            iy = neigh[:,1] % grid_size
            iz = neigh[:,2] % grid_size

            np.add.at(vel_x, (ix, iy, iz), velocities[:,0] * w)
            np.add.at(vel_y, (ix, iy, iz), velocities[:,1] * w)
            np.add.at(vel_z, (ix, iy, iz), velocities[:,2] * w)
            np.add.at(count, (ix, iy, iz), w)

    elif method == "cic":
        # Cloud In Cell (8 neighbors)
        scaled_pos = pos_frac * grid_size
        base = np.floor(scaled_pos).astype(int)
        frac = scaled_pos - base

        for dx in (0, 1):
            wx = (1 - frac[:, 0]) if dx == 0 else frac[:, 0]
            ix = (base[:, 0] + dx) % grid_size
            for dy in (0, 1):
                wy = (1 - frac[:, 1]) if dy == 0 else frac[:, 1]
                iy = (base[:, 1] + dy) % grid_size
                for dz in (0, 1):
                    wz = (1 - frac[:, 2]) if dz == 0 else frac[:, 2]
                    iz = (base[:, 2] + dz) % grid_size

                    w = wx * wy * wz
                    np.add.at(vel_x, (ix, iy, iz), velocities[:, 0] * w)
                    np.add.at(vel_y, (ix, iy, iz), velocities[:, 1] * w)
                    np.add.at(vel_z, (ix, iy, iz), velocities[:, 2] * w)
                    np.add.at(count, (ix, iy, iz), w)
    else:
        raise ValueError(f"Unknown method '{method}', use 'ngp', 'tsc' or 'cic'.")

    return vel_x, vel_y, vel_z, count

def build_mass_grid(positions: np.ndarray,
                    masses: np.ndarray,
                    box_size: float,
                    grid_size: int = 100,
                    method: str = "ngp",
                    mass_grid:np.ndarray = None):
    """
    Bin particle masses into a 3D grid with different assignment schemes.

    Parameters
    ----------
    positions : (N, 3) array
        Particle positions, assumed inside [0, box_size).
    masses : (N,) array
        Particle masses.
    box_size : float
        Size of the simulation box (same units as positions).
    grid_size : int
        Number of cells per dimension (default 100).
    method : str, optional
        Interpolation scheme:
          - "ngp": Nearest Grid Point (default)
          - "cic": Cloud In Cell
          - "tsc": Triangular Shaped Cloud
    mass_grid : np.ndarray, optional
        If no grid is passed in, it will construct a new one.

    Returns
    -------
    mass_grid : (grid_size, grid_size, grid_size) array
        Mass in each cell.
    count_grid : (grid_size, grid_size, grid_size) array
        Total weight assigned to each cell (useful for normalization).
    """

    if mass_grid is None:
        vel_x = np.zeros((grid_size, grid_size, grid_size), dtype=float)
        vel_y = np.zeros_like(vel_x)
        vel_z = np.zeros_like(vel_x)
        count = np.zeros_like(vel_x)

    pos_frac = positions / box_size
    scaled_pos = pos_frac * grid_size

    if method == "ngp":
        # Nearest Grid Point = closest cell center
        indices = np.floor(scaled_pos + 0.5).astype(int) % grid_size
        np.add.at(mass_grid, tuple(indices.T), masses)

    elif method == "cic":
        # Cloud In Cell (8 neighbors)
        base = np.floor(scaled_pos).astype(int)
        frac = scaled_pos - base

        for dx in (0, 1):
            wx = (1 - frac[:, 0]) if dx == 0 else frac[:, 0]
            ix = (base[:, 0] + dx) % grid_size
            for dy in (0, 1):
                wy = (1 - frac[:, 1]) if dy == 0 else frac[:, 1]
                iy = (base[:, 1] + dy) % grid_size
                for dz in (0, 1):
                    wz = (1 - frac[:, 2]) if dz == 0 else frac[:, 2]
                    iz = (base[:, 2] + dz) % grid_size

                    w = wx * wy * wz
                    np.add.at(mass_grid, (ix, iy, iz), masses * w)

    elif method == "tsc":
        # Triangular Shaped Cloud (27 neighbors)
        def tsc_kernel(x):
            ax = np.abs(x)
            w = np.zeros_like(ax)
            m1 = ax < 0.5
            w[m1] = 0.75 - ax[m1]**2
            m2 = (ax >= 0.5) & (ax < 1.5)
            w[m2] = 0.5 * (1.5 - ax[m2])**2
            return w
        
        base = np.floor(scaled_pos).astype(int)
        shifts = np.array([[-1,-1,-1], [-1,-1,0], [-1,-1,1],
                           [-1, 0,-1], [-1, 0,0], [-1, 0,1],
                           [-1, 1,-1], [-1, 1,0], [-1, 1,1],
                           [ 0,-1,-1], [ 0,-1,0], [ 0,-1,1],
                           [ 0, 0,-1], [ 0, 0,0], [ 0, 0,1],
                           [ 0, 1,-1], [ 0, 1,0], [ 0, 1,1],
                           [ 1,-1,-1], [ 1,-1,0], [ 1,-1,1],
                           [ 1, 0,-1], [ 1, 0,0], [ 1, 0,1],
                           [ 1, 1,-1], [ 1, 1,0], [ 1, 1,1]], dtype=int)

        for sx, sy, sz in shifts:
            neigh = (base + (sx, sy, sz)) % grid_size  

            d = scaled_pos - (neigh + 0.5)
            d = np.where(d >  grid_size/2, d - grid_size, d)
            d = np.where(d < -grid_size/2, d + grid_size, d)

            w = tsc_kernel(d[:, 0]) * tsc_kernel(d[:, 1]) * tsc_kernel(d[:, 2])

            np.add.at(mass_grid, (neigh[:, 0], neigh[:, 1], neigh[:, 2]), masses * w)

    else:
        raise ValueError(f"Unknown method '{method}', use 'ngp', 'cic' or 'tsc'.")
    

    return mass_grid

def compute_shear_tensor(vel_x, vel_y, vel_z, box_size, H0=70.0):
    """
    Compute Sigma_{alpha,beta} = -1/(2 H0) (d_beta v_alpha + d_alpha v_beta)
    using FFT derivatives (periodic box).

    Inputs
    ------
    vel_x, vel_y, vel_z : (N,N,N) real arrays
        Velocity component grids (should be average velocity per cell if you want velocity shear).
    box_size : float
        Physical box length (same units as distances used to make velocity grid).
    H0 : float
        Hubble constant in velocity/distance units (e.g. km/s/Mpc). Default 70.0.

    Returns
    -------
    sigma : dict with keys ('xx','yy','zz','xy','xz','yz') each an (N,N,N) array
        Independent components of Sigma (note sigma['xy'] == sigma['yx']).
    """ 

    def safe_ifft(arr_k, kcomp):
        """Take derivative in Fourier space and check imaginary part."""
        arr = np.fft.ifftn(1j * kcomp * arr_k)
        im_max = np.max(np.abs(arr.imag))
        re_max = np.max(np.abs(arr.real))
        #if im_max > 1e-10 * (re_max + 1e-30):
            #print(f"Warning: large imaginary part (max {im_max:.3e}, max real {re_max:.3e})")
        
        return arr.real

    N = vel_x.shape[0]
    assert vel_x.shape == vel_y.shape == vel_z.shape == (N, N, N)
    dx = box_size / N

    freqs = np.fft.fftfreq(N, d=dx)    
    #k1d = 2.0 * np.pi * freqs        
    k1d = 2.0 * np.pi * freqs    
    kx = k1d[:, None, None]
    ky = k1d[None, :, None]
    kz = k1d[None, None, :]

    vx_k = np.fft.fftn(vel_x)
    
    vx_k = np.fft.fftn(vel_x)
    vy_k = np.fft.fftn(vel_y)
    vz_k = np.fft.fftn(vel_z)


    # Derivatives
    d_vx_dx = safe_ifft(vx_k, kx)
    d_vy_dx = safe_ifft(vy_k, kx)
    d_vz_dx = safe_ifft(vz_k, kx)

    d_vx_dy = safe_ifft(vx_k, ky)
    d_vy_dy = safe_ifft(vy_k, ky)
    d_vz_dy = safe_ifft(vz_k, ky)

    d_vx_dz = safe_ifft(vx_k, kz)
    d_vy_dz = safe_ifft(vy_k, kz)
    d_vz_dz = safe_ifft(vz_k, kz)

    S_xx = 0.5 * (d_vx_dx + d_vx_dx) 
    S_yy = 0.5 * (d_vy_dy + d_vy_dy)
    S_zz = 0.5 * (d_vz_dz + d_vz_dz)
    S_xy = 0.5 * (d_vx_dy + d_vy_dx)
    S_xz = 0.5 * (d_vx_dz + d_vz_dx)
    S_yz = 0.5 * (d_vy_dz + d_vz_dy)

    inv_H0 = 1.0 / float(H0)
    sigma = {
        'xx': - S_xx * inv_H0,
        'yy': - S_yy * inv_H0,
        'zz': - S_zz * inv_H0,
        'xy': - S_xy * inv_H0,
        'xz': - S_xz * inv_H0,
        'yz': - S_yz * inv_H0,
    }

    return sigma

def compute_average_velocity(vel_x, vel_y, vel_z, count, smooting_fine=1):
    mask = count > 0.0
    avg_vx = np.zeros_like(vel_x)
    avg_vy = np.zeros_like(vel_y)
    avg_vz = np.zeros_like(vel_z)
  
    avg_vx[mask] = vel_x[mask] / count[mask]
    avg_vy[mask] = vel_y[mask] / count[mask]
    avg_vz[mask] = vel_z[mask] / count[mask]
  
    avg_vx = gaussian_filter(avg_vx, sigma=smoothing_fine, mode="wrap")
    avg_vy = gaussian_filter(avg_vy, sigma=smoothing_fine, mode="wrap")
    avg_vz = gaussian_filter(avg_vz, sigma=smoothing_fine, mode="wrap")
  
    return avg_vx, avg_vy, avg_vz
  
def compute_density_grid(mass_grid,box_size,grid_size,smoothing_fine=1):
    density_grid = mass_grid / (box_size / grid_size) ** 3
    density_grid /= np.mean(density_grid)
    return gaussian_filter(density_grid, sigma=smoothing_fine, mode="wrap")


def diagonalize_shear_tensor(sigma):
    """
    Compute eigenvalues and eigenvectors of the velocity shear tensor
    at each cell in a 3D grid in a fully vectorized way.

    Parameters
    ----------
    sigma : dict
        Dictionary with keys ('xx','yy','zz','xy','xz','yz') each an (N,N,N) array.

    Returns
    -------
    lambdas : tuple of arrays
        (lambda1, lambda2, lambda3), each (N,N,N), sorted descending: lambda1 > lambda2 > lambda3
    evecs : tuple of arrays
        (evec1, evec2, evec3), each (N,N,N,3), corresponding eigenvectors.
    """
    N = sigma['xx'].shape[0]
    
    tensor = np.zeros((N, N, N, 3, 3), dtype=np.float32)
    tensor[..., 0, 0] = sigma['xx']
    tensor[..., 1, 1] = sigma['yy']
    tensor[..., 2, 2] = sigma['zz']
    tensor[..., 0, 1] = tensor[..., 1, 0] = sigma['xy']
    tensor[..., 0, 2] = tensor[..., 2, 0] = sigma['xz']
    tensor[..., 1, 2] = tensor[..., 2, 1] = sigma['yz']

    tensor_flat = tensor.reshape(-1, 3, 3)

    vals, vecs = np.linalg.eigh(tensor_flat) 

    idx = np.argsort(vals, axis=1)[:, ::-1]  
    vals_sorted = np.take_along_axis(vals, idx, axis=1)
    #vecs_sorted = np.take_along_axis(vecs, idx[:, None, :], axis=2)

    lambda1 = vals_sorted[:, 0].reshape(N, N, N)
    lambda2 = vals_sorted[:, 1].reshape(N, N, N)
    lambda3 = vals_sorted[:, 2].reshape(N, N, N)
    #evec1 = vecs_sorted[:, :, 0].reshape(N, N, N, 3)
    #evec2 = vecs_sorted[:, :, 1].reshape(N, N, N, 3)
    #evec3 = vecs_sorted[:, :, 2].reshape(N, N, N, 3)

    return (lambda1, lambda2, lambda3)#, (evec1, evec2, evec3)

def classify_cosmic_web(lambdas, lam_th=0.0):
    """
    Classify cosmic web based on shear tensor eigenvalues.

    Parameters
    ----------
    lambdas : tuple of arrays
        (lambda1, lambda2, lambda3), each shape (N,N,N)
    lam_th : float
        Threshold eigenvalue

    Returns
    -------
    web_type : array
        Integer array (N,N,N) with:
        0 = void, 1 = sheet, 2 = filament, 3 = cluster
    """
    lambda1, lambda2, lambda3 = lambdas

    # Count eigenvalues above threshold
    count = (lambda1 > lam_th).astype(int) + \
            (lambda2 > lam_th).astype(int) + \
            (lambda3 > lam_th).astype(int)

    # Map to cosmic web type: 0=void, 1=sheet, 2=filament, 3=cluster
    web_type = count  
    
    return web_type

def apply_multiscale_correction(fine_web, coarse_web, density_grid, 
                               mean_density=1.0, virial_density=340.0):
    """
    Apply multiscale correction to the V-web classification as described in 
    Hoffman et al. (2012) Section 5.
    
    Parameters
    ----------
    fine_web : array_like, shape (N,N,N)
        High-resolution cosmic web classification (0=void, 1=sheet, 2=filament, 3=knot)
    coarse_web : array_like, shape (N,N,N)
        Low-resolution cosmic web classification with same shape and encoding
    density_grid : array_like, shape (N,N,N)
        Density field in units where mean density = 1.0
    mean_density : float, default 1.0
        Mean density threshold (normalized)
    virial_density : float, default 340.0
        Virial density threshold for correction
        
    Returns
    -------
    corrected_web : array_like, shape (N,N,N)
        Multiscale-corrected cosmic web classification
        
    Notes
    -----
    This function corrects two types of misclassifications:
    1. Voids (fine_web=0) with overdensity >= mean_density
    2. Sheets/filaments (fine_web=1,2) with overdensity >= virial_density
    
    Both are replaced with the coarse web classification at the same location.
    """
    corrected_web = fine_web.copy()
    N = fine_web.shape[0]
    
    corrections_applied = 0
    
    for ix in range(N):
        for iy in range(N):
            for iz in range(N):
                dens = density_grid[ix, iy, iz]
                fine_val = fine_web[ix, iy, iz]
                
                # Correction 1: Voids with overdensity >= mean density
                if fine_val == 0 and dens >= mean_density:
                    corrected_web[ix, iy, iz] = coarse_web[ix, iy, iz]
                    corrections_applied += 1
                    
                # Correction 2: Sheets/filaments with overdensity >= virial density
                elif fine_val in [1, 2] and dens >= virial_density:
                    corrected_web[ix, iy, iz] = coarse_web[ix, iy, iz]
                    corrections_applied += 1
    
    print(f"Multiscale correction applied to {corrections_applied} cells")
    return corrected_web

def plotting_routine(web,box_size,grid_size,threshold):
    """
    Plotting routine for the cosmic web classification.
    
    Parameters
    ----------
    web : (N,N,N) array
        Cosmic web classification grid.
    box_size : float
        Size of the simulation box (same units as positions)
    grid_size : int
        Number of cells per dimension
    threshold : float
        Eigenvalue threshold used in classification
    
    Returns
    -------
    None
    """
    print("Done. Plotting...")

    # Parameters
    z_mid_idx = web.shape[2] // 2   # middle z index
    dz = (box_size / web.shape[2]) / 2  # half cell width in z
    cmap = ListedColormap(['white', 'cyan', 'orange', 'red'])
    labels = ['Void', 'Sheet', 'Filament', 'Cluster']

    # Get the slice of the web grid
    web_slice = web[:, :, z_mid_idx]

    # Plot the grid-colored cosmic web
    extent = [0, box_size, 0, box_size]

    plt.figure(figsize=(16, 16))
    im = plt.imshow(web_slice.T, origin='lower', extent=extent, cmap=cmap, interpolation='nearest')

    plt.xlim(0, box_size)
    plt.ylim(0, box_size)

    plt.xlabel('X [Mpc]')
    plt.ylabel('Y [Mpc]')
    plt.title('Cosmic Web with Multiscale Correction (Middle Z slice)')
    plt.grid(color='gray', linestyle='--', alpha=0.5)
    
    
class CosmicWebClassifier:
    def __init__(self, 
                 box_size: float = 100.0, 
                 grid_size: int = 256, 
                 method: str = "cic", 
                 threshold: float = 0.44,
                 H0: float = 67.5,
                 smoothing_fine: float = 1,
                 smoothing_coarse: float = 4):
        self.box_size = box_size
        self.grid_size = grid_size
        self.method = method
        self.threshold = threshold
        self.H0 = H0
        self.smoothing_fine = smoothing_fine
        self.smoothing_coarse = smoothing_coarse

        self.reset_grids()

        self.web = None
        self.evecs = None
        self.lambdas_fine = None
        self.lambdas_coarse = None


    def reset_grids(self):
        shape = (self.grid_size, self.grid_size, self.grid_size)
        self.vel_x = np.zeros(shape, dtype=np.float64)
        self.vel_y = np.zeros(shape, dtype=np.float64)
        self.vel_z = np.zeros(shape, dtype=np.float64)
        self.count = np.zeros(shape, dtype=np.float64)
        self.mass_grid = np.zeros(shape, dtype=np.float64)

    def add_batch(self, positions, velocities, masses=None):

        if masses is None:
            masses = np.ones(len(positions), dtype=np.float64)

        """self.vel_x, self.vel_y, self.vel_z, self.count = build_velocity_grid(
            positions, velocities, self.box_size, 
            grid_size=self.grid_size, method=self.method,
            vel_x=self.vel_x, vel_y=self.vel_y, vel_z=self.vel_z, count=self.count
        )
        self.mass_grid = build_mass_grid(
            positions, masses, self.box_size, 
            grid_size=self.grid_size, method=self.method, mass_grid=self.mass_grid
        )"""

        self.vel_x, self.vel_y, self.vel_z, self.count = build_velocity_grid_numba(
                  positions, velocities, self.box_size, 
                  grid_size=self.grid_size, method=self.method,
                  vel_x=self.vel_x, vel_y=self.vel_y, vel_z=self.vel_z, count=self.count
              )
        self.mass_grid = build_mass_grid_numba(
            positions, masses, self.box_size, 
            grid_size=self.grid_size, method=self.method, mass_grid=self.mass_grid
        )

    # ------------------------
    # Final computation
    # ------------------------
    def classify_structure(self):

        avg_vx, avg_vy, avg_vz = self._compute_average_velocity()
        density_grid = self._compute_density_grid()

        sigma_fine = compute_shear_tensor(avg_vx, avg_vy, avg_vz, box_size=self.box_size, H0=self.H0)
        self.lambdas_fine = diagonalize_shear_tensor(sigma_fine)
        web_fine = classify_cosmic_web(self.lambdas_fine, lam_th=self.threshold)

        sigma_coarse = compute_shear_tensor(
            gaussian_filter(avg_vx, sigma=self.smoothing_coarse, mode='wrap'),
            gaussian_filter(avg_vy, sigma=self.smoothing_coarse, mode='wrap'),
            gaussian_filter(avg_vz, sigma=self.smoothing_coarse, mode='wrap'),
            box_size=self.box_size, H0=self.H0
        )

        self.lambdas_coarse= diagonalize_shear_tensor(sigma_coarse)
        web_coarse = classify_cosmic_web(self.lambdas_coarse, lam_th=self.threshold)

        self.web = apply_multiscale_correction(
            web_fine, web_coarse, density_grid, mean_density=1.0, virial_density=340.0
        )
        return self.web


    def plot(self, filename=None,show=False):
        if self.web is None:
            raise RuntimeError("Run classify_structure() first after adding batches.")

        plotting_routine(self.web, self.box_size, self.grid_size, self.threshold)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


    def _compute_average_velocity(self):
        mask = self.count > 0.0
        avg_vx = np.zeros_like(self.vel_x)
        avg_vy = np.zeros_like(self.vel_y)
        avg_vz = np.zeros_like(self.vel_z)
        avg_vx[mask] = self.vel_x[mask] / self.count[mask]
        avg_vy[mask] = self.vel_y[mask] / self.count[mask]
        avg_vz[mask] = self.vel_z[mask] / self.count[mask]

        avg_vx = gaussian_filter(avg_vx, sigma=self.smoothing_fine, mode="wrap")
        avg_vy = gaussian_filter(avg_vy, sigma=self.smoothing_fine, mode="wrap")
        avg_vz = gaussian_filter(avg_vz, sigma=self.smoothing_fine, mode="wrap")
        return avg_vx, avg_vy, avg_vz

    def _compute_density_grid(self):
        density_grid = self.mass_grid / (self.box_size / self.grid_size) ** 3
        density_grid /= np.mean(density_grid)
        return gaussian_filter(density_grid, sigma=self.smoothing_fine, mode="wrap")






