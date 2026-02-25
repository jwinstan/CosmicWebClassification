import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from numba import njit, prange
import functools
import inspect
import sys
import psutil
import os
import warnings
from mpi4py import MPI




def sizeof_var(var):
    """Return memory usage of a variable in bytes."""
    if isinstance(var, np.ndarray):
        return var.nbytes
    return sys.getsizeof(var)

def report_locals_memory(frame, label=""):
    """Report memory usage of all local variables in a frame."""
    print(f"\n[Memory Report] {label}")
    total = 0
    for name, value in frame.f_locals.items():
        try:
            size = sizeof_var(value)
        except Exception:
            size = 0
        total += size
        print(f"{name:20s} : {size/1024/1024:.3f} MB")
    print(f"{'-'*35}\nTotal locals: {total/1024/1024:.3f} MB")

    # also report process RSS
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    print(f"Process RSS   : {rss/1024/1024:.3f} MB")

def memory_profile(label=""):
    """
    Decorator to profile memory usage of local variables
    and process RSS for a function.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            frame = inspect.currentframe().f_back
            report_locals_memory(frame, label=f"{func.__name__} {label}")
            return result
        return wrapper
    return decorator

def memory_profile_class(cls):
    """
    Class decorator: wrap all methods to profile memory.
    """
    for name, attr in list(cls.__dict__.items()):
        if callable(attr):
            setattr(cls, name, memory_profile()(attr))
    return cls



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

#@memory_profile()
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

#@memory_profile()
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

#@memory_profile()
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
    if vel_x.shape != vel_y.shape or vel_x.shape != vel_z.shape:
        raise ValueError("Velocity grids must have the same shape")
    if vel_x.shape[0] != vel_x.shape[1] or vel_x.shape[1] != vel_x.shape[2]:
        raise ValueError("Velocity grids must be cubic (N,N,N)")
    
    if not np.all(np.isfinite(vel_x)) or not np.all(np.isfinite(vel_y)) or not np.all(np.isfinite(vel_z)):
        raise ValueError("Velocity grids contain NaNs or infs")
    
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

    S_xx = d_vx_dx
    S_yy = d_vy_dy
    S_zz = d_vz_dz
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

#@memory_profile()
def compute_average_velocity(vel_x, vel_y, vel_z, count, smoothing_fine=1):
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

#@memory_profile()
def compute_density_grid(mass_grid,box_size,grid_size,smoothing_fine=1):
    density_grid = mass_grid / (box_size / grid_size) ** 3
    density_grid /= np.mean(density_grid)
    return gaussian_filter(density_grid, sigma=smoothing_fine, mode="wrap")

#@memory_profile()
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

    try:
        vals, vecs = np.linalg.eigh(tensor_flat)
    except np.linalg.LinAlgError as e:
        raise RuntimeError("Eigen-decomposition failed. Check the shear tensor values.") from e

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

def diagonalize_shear_tensor_new(sigma):
    """
    Compute eigenvalues of the velocity shear tensor at each cell in a 3D grid.

    Parameters
    ----------
    sigma : dict
        Dictionary with keys ('xx','yy','zz','xy','xz','yz') each an (N,N,N) array.

    Returns
    -------
    lambdas : tuple of arrays
        (lambda1, lambda2, lambda3), each (N,N,N), sorted descending: lambda1 > lambda2 > lambda3
    """
    N = sigma['xx'].shape[0]
    
    # Build 3x3 tensor at each grid cell
    tensor = np.zeros((N, N, N, 3, 3), dtype=np.float32)
    tensor[..., 0, 0] = sigma['xx']
    tensor[..., 1, 1] = sigma['yy']
    tensor[..., 2, 2] = sigma['zz']
    tensor[..., 0, 1] = tensor[..., 1, 0] = sigma['xy']
    tensor[..., 0, 2] = tensor[..., 2, 0] = sigma['xz']
    tensor[..., 1, 2] = tensor[..., 2, 1] = sigma['yz']

    # Compute eigenvalues
    vals = np.linalg.eigh(tensor)[0]  # shape: (N,N,N,3), ascending order

    # Sort descending
    vals_sorted = vals[..., ::-1]

    lambda1 = vals_sorted[..., 0]
    lambda2 = vals_sorted[..., 1]
    lambda3 = vals_sorted[..., 2]

    return (lambda1, lambda2, lambda3)


#@memory_profile()
def classify_cosmic_web(lambdas, lam_th=0.44):
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

#@memory_profile()
def apply_multiscale_correction_old(fine_web, coarse_web, density_grid, 
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

    assert fine_web.shape == coarse_web.shape == density_grid.shape, "Input grids must have the same shape."
    
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
    
    assert fine_web.shape == coarse_web.shape == density_grid.shape, \
        "Input grids must have the same shape."
    
    # Correction 1: Voids with overdensity >= mean density
    mask1 = (fine_web == 0) & (density_grid >= mean_density)
    corrected_web[mask1] = coarse_web[mask1]
    corrections_1 = np.sum(mask1)

    # Correction 2: Sheets/filaments with overdensity >= virial density
    mask2 = np.isin(fine_web, [1, 2]) & (density_grid >= virial_density)
    corrected_web[mask2] = coarse_web[mask2]
    corrections_2 = np.sum(mask2)
    
    corrections_applied = corrections_1 + corrections_2
    print(f"Multiscale correction applied to {corrections_applied} cells "
          f"({corrections_1} voids, {corrections_2} sheets/filaments)")
    
    return corrected_web

#@memory_profile()
def plotting_routine(web,box_size,grid_size,threshold,z_level=None):
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

    if not np.all(np.isin(web, [0,1,2,3])):
        print("Warning: web contains unexpected values outside [0,1,2,3]")

    # Parameters
    if z_level == None:
        z_mid_idx = web.shape[2] // 2   # middle z index
    else:
        assert 0 <= z_level < web.shape[2], "z_level must be within the grid range"
        z_mid_idx = int(z_level)

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
    plt.title(f'Cosmic Web ({z_mid_idx} slice)')
    plt.grid(color='gray', linestyle='--', alpha=0.5)
    
#@memory_profile_class
class CosmicWebClassifier:
    def __init__(self,
                 box_size: float = 100.0,
                 grid_size: int = 256,
                 method: str = "cic",
                 threshold: float = 0.44,
                 H0: float = 67.5,
                 smoothing_fine: float = 0.25,
                 smoothing_coarse: float = 1.0,
                 smoothing_units: str = "physical",
                 apply_multiscale_correction: bool = True):
        """
        Parameters
        ----------
        box_size : float
            Simulation box size in Mpc/h.
        grid_size : int
            Number of cells along one axis.
        method : {"ngp","cic","tsc"}
            Mass / velocity assignment scheme.
        threshold : float
            Eigenvalue threshold (lam_th).
        H0 : float
            Hubble constant in km/s/(Mpc/h).
        smoothing_fine : float
            Fine-scale Gaussian smoothing length.
            Interpreted either in physical units (Mpc/h)
            or in grid-cell units, controlled by `smoothing_units`.
        smoothing_coarse : float
            Coarse-scale smoothing length (same units as above).
        smoothing_units : {"physical","cells"}
            - "physical": input smoothing_* are in Mpc/h,
              they will be scaled to grid cells as
              sigma_cells = smoothing * grid_size / box_size.
            - "cells": input smoothing_* are already in grid-cell units
              and used as-is (no scaling).
        apply_multiscale_correction : bool
            Whether to apply multiscale correction step.
        """

        self.box_size = float(box_size)
        self.grid_size = int(grid_size)
        self.method = method
        self.threshold = float(threshold)
        self.H0 = float(H0)

        #Quick memory check
        available_mem = psutil.virtual_memory().available  # bytes
        n_cells = self.grid_size ** 3
        bytes_per_element = np.dtype(np.float64).itemsize  # 8 bytes
        estimated_arrays = 1 + 1 + 3 + 3 + 9
        estimated_mem = n_cells * bytes_per_element * estimated_arrays
        estimated_mem *= 1.3

        if estimated_mem > 0.8 * available_mem:
            warnings.warn(
                f"Estimated memory usage ≈ {estimated_mem/1e9:.2f} GB, "
                f"available system memory ≈ {available_mem/1e9:.2f} GB.\n"
                "Job might not finish: possible insufficient memory.",
                ResourceWarning
            )

        if smoothing_units not in ("physical", "cells"):
            raise ValueError("smoothing_units must be 'physical' or 'cells'")

        if smoothing_units == "physical":
            self.smoothing_fine   = smoothing_fine   * self.grid_size / self.box_size
            self.smoothing_coarse = smoothing_coarse * self.grid_size / self.box_size
        else:  # "cells"
            self.smoothing_fine   = float(smoothing_fine)
            self.smoothing_coarse = float(smoothing_coarse)

        self.msc = bool(apply_multiscale_correction)
          
        if self.box_size <= 0:
            raise ValueError(f"box_size must be positive, got {self.box_size}")
        if self.grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {self.grid_size}")
        
        valid_methods = ["ngp", "cic", "tsc"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{method}', must be one of {valid_methods}")

        self.reset_grids()

        self.web = None
        self.evecs = None

    def reset_grids(self):
        shape = (self.grid_size, self.grid_size, self.grid_size)
        try:
            self.vel_x = np.zeros(shape, dtype=np.float64)
            self.vel_y = np.zeros(shape, dtype=np.float64)
            self.vel_z = np.zeros(shape, dtype=np.float64)
        except MemoryError as e:
            raise MemoryError(f"Failed to allocate velocity grids of shape {shape}. Try reducing grid_size.") from e
        try:
            self.count = np.zeros(shape, dtype=np.float64)
        except MemoryError as e:
            raise MemoryError(f"Failed to allocate count grid of shape {shape}. Try reducing grid_size.") from e
        try:
            self.mass_grid = np.zeros(shape, dtype=np.float64)
        except MemoryError as e:
            raise MemoryError(f"Failed to allocate mass grid of shape {shape}. Try reducing grid_size.") from e

    def add_batch(self, positions, velocities, masses=None):
        self._validate_particle_inputs(positions, velocities, masses)
        assert np.all(positions >= 0.0) and np.all(positions <= self.box_size), "Positions must be within the box [0, box_size)."
        pos_max = np.max(positions)
        if pos_max < 0.9*self.box_size:
            print(f"Warning: max position {pos_max:.3f} is much smaller than box_size {self.box_size:.3f}.")
        if masses is None:
            masses = np.ones(positions.shape[0], dtype=np.float64)
        build_velocity_grid_numba(
                  positions, velocities, self.box_size, 
                  grid_size=self.grid_size, method=self.method,
                  vel_x=self.vel_x, vel_y=self.vel_y, vel_z=self.vel_z, count=self.count
              )
        build_mass_grid_numba(
            positions, masses, self.box_size, 
            grid_size=self.grid_size, method=self.method, mass_grid=self.mass_grid
        )

        del positions, velocities, masses

    # ------------------------
    # Final computation
    # ------------------------
    def classify_structure(self):
        avg_vx, avg_vy, avg_vz = self._compute_average_velocity()
        density_grid = self._compute_density_grid()

        sigma_fine = compute_shear_tensor(avg_vx, avg_vy, avg_vz, box_size=self.box_size, H0=self.H0)
        lambdas_fine = diagonalize_shear_tensor(sigma_fine)
        web_fine = classify_cosmic_web(lambdas_fine, lam_th=self.threshold)  

        if self.msc:
            sigma_coarse = compute_shear_tensor(
            gaussian_filter(avg_vx, sigma=self.smoothing_coarse, mode='wrap'),
            gaussian_filter(avg_vy, sigma=self.smoothing_coarse, mode='wrap'),
            gaussian_filter(avg_vz, sigma=self.smoothing_coarse, mode='wrap'),
            box_size=self.box_size, H0=self.H0
            )
    
            lambdas_coarse=  diagonalize_shear_tensor(sigma_coarse)
            web_coarse = classify_cosmic_web(lambdas_coarse, lam_th=self.threshold)
            self.web = apply_multiscale_correction(web_fine, web_coarse, density_grid, mean_density=1.0, virial_density=340.0)     
        else:
            self.web = web_fine
        return self.web

    def plot(self, filename=None,show=True,z_level=None):
        if self.web is None:
            raise RuntimeError("Run classify_structure() first after adding batches.")

        plotting_routine(self.web, self.box_size, self.grid_size, self.threshold,z_level=z_level)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        else:
            if show:
                plt.show()
    
    def update_threshold(self,threshold_updated):
        if self.web is None:
            raise RuntimeError("Run classify_structure() first after adding batches. This is to update the threshold and re-run without re-computing velocity grids.")
        
        avg_vx, avg_vy, avg_vz = self._compute_average_velocity()
        density_grid = self._compute_density_grid()

        sigma_fine = compute_shear_tensor(avg_vx, avg_vy, avg_vz, box_size=self.box_size, H0=self.H0)
        lambdas_fine = diagonalize_shear_tensor(sigma_fine)
        web_fine = classify_cosmic_web(lambdas_fine, lam_th=threshold_updated)
        if self.msc:
            sigma_coarse = compute_shear_tensor(
            gaussian_filter(avg_vx, sigma=self.smoothing_coarse, mode='wrap'),
            gaussian_filter(avg_vy, sigma=self.smoothing_coarse, mode='wrap'),
            gaussian_filter(avg_vz, sigma=self.smoothing_coarse, mode='wrap'),
            box_size=self.box_size, H0=self.H0
            )
    
            lambdas_coarse=  diagonalize_shear_tensor(sigma_coarse)
            web_coarse = classify_cosmic_web(lambdas_coarse, lam_th=threshold_updated)
            self.web = apply_multiscale_correction(web_fine, web_coarse, density_grid, mean_density=1.0, virial_density=340.0)     
        else:
            self.web = web_fine
        return self.web
        
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
    
    def _validate_particle_inputs(self, positions, velocities, masses=None):
        if not isinstance(positions, np.ndarray) or not isinstance(velocities, np.ndarray):
            raise TypeError("positions and velocities must be numpy arrays")
        if positions.shape != velocities.shape:
            raise ValueError(f"positions and velocities must have the same shape, got {positions.shape} and {velocities.shape}")
        if positions.shape[1] != 3:
            raise ValueError(f"positions and velocities must have shape (N,3), got {positions.shape}")
        if masses is not None:
            if not isinstance(masses, np.ndarray):
                raise TypeError("masses must be a numpy array")
            if masses.shape[0] != positions.shape[0]:
                raise ValueError(f"masses must have length {positions.shape[0]}, got {masses.shape[0]}")
        # Check for NaNs/Infs
        if not np.all(np.isfinite(positions)):
            raise ValueError("positions contain NaNs or infinite values")
        if not np.all(np.isfinite(velocities)):
            raise ValueError("velocities contain NaNs or infinite values")
        if masses is not None and not np.all(np.isfinite(masses)):
            raise ValueError("masses contain NaNs or infinite values")


    def _get_mpi(self):
        try:
            from mpi4py import MPI
        except Exception:
            return None, 0, 1, None
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size(), MPI

    def mpi_reduce_all_grids_inplace(self, root: int = 0, everyone: bool = False) -> bool:
        """
        Reduce ALL grids in-place to minimize RAM.

        - everyone=False: Reduce to `root` only. Only root returns True.
        - everyone=True : Allreduce in-place. Everyone returns True.

        No full-sized temporary recv arrays are allocated.
        """
        comm, rank, size, MPI = self._get_mpi()
        if comm is None or size <= 1:
            return True

        grids = (self.vel_x, self.vel_y, self.vel_z, self.count, self.mass_grid)

        if everyone:
            # In-place Allreduce: sums into the existing buffers on all ranks
            for g in grids:
                comm.Allreduce(MPI.IN_PLACE, g, op=MPI.SUM)
            return True

        # Root-only in-place Reduce
        if rank == root:
            for g in grids:
                comm.Reduce(MPI.IN_PLACE, g, op=MPI.SUM, root=root)
            return True
        else:
            for g in grids:
                comm.Reduce(g, None, op=MPI.SUM, root=root)
            return False
        


















