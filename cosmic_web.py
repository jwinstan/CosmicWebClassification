import numpy as np
from scipy.ndimage import gaussian_filter
import gc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def cosmic_web_classification_routine():
    """
    A routine to classify the cosmic web from Hoffman et al 2012
    link: https://academic.oup.com/mnras/article/425/3/2049/982860 
    """
    grid_size = 256
    box_size = 100.0 #hard coded for now, can extract from simulation file.
    method = "cic" #I recommend cic right now, tsc has some wacky stuff going on (but it still works)
    threshold = 0.44 #Free parameter. Change until the universe looks good.


    #Positions need to be in a (N,3) array
    #Same array structure for velocities
    #(N,1) array for masses

    #Pre-construction of the arrays
    vel_x = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
    vel_y = np.zeros_like(vel_x)
    vel_z = np.zeros_like(vel_x)
    count = np.zeros_like(vel_x)
    mass_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)

    """
    This is where data processing goes, this can be in a loop over multiple files, just contain the build_velocity_grid call in the loop.
    If all masses the same, and array of 1s can be used, since all that matters is density.
    for file in files:
        positions, velocities, masses = load_file(file)
        vel_x, vel_y, vel_z, count = build_velocity_grid(vel_x, vel_y, vel_z, count, positions, velocities, box_size, grid_size=grid_size, method=method)
    """

    #Temporary random data for testing
    N = 100_000
    positions = np.random.rand(N, 3) * box_size
    velocities = np.random.randn(N, 3)  # can be anything
    masses = np.ones(N, dtype= np.float64)

    print("Building velocity and density grids...")
    vel_x, vel_y, vel_z, count = build_velocity_grid(positions, velocities, box_size, grid_size=grid_size, method=method, 
                                                     vel_x=vel_x, vel_y=vel_y, vel_z=vel_z, count=count)
    mass_grid = build_mass_grid(positions, masses, box_size, grid_size=grid_size, method=method,mass_grid=mass_grid)

    mask = count > 0.0
    # avoid division-by-zero; set zeros where no particles
    avg_vx = np.zeros_like(vel_x, dtype=np.float64)
    avg_vy = np.zeros_like(vel_y, dtype=np.float64)
    avg_vz = np.zeros_like(vel_z, dtype=np.float64)
    avg_vx[mask] = vel_x[mask] / count[mask]
    avg_vy[mask] = vel_y[mask] / count[mask]
    avg_vz[mask] = vel_z[mask] / count[mask]

    vel_x = gaussian_filter(avg_vx, sigma=1, mode="wrap")
    vel_y = gaussian_filter(avg_vy, sigma=1, mode="wrap")
    vel_z = gaussian_filter(avg_vz, sigma=1, mode="wrap")
   
    
    density_grid = mass_grid / (box_size / grid_size)**3  # convert to density
    # Normalize to mean density = 1
    density_grid = density_grid / np.mean(density_grid)
    density_grid = gaussian_filter(density_grid, sigma=1, mode="wrap")
    gc.collect()

    print("Computing fine shear tensor...")
    sigma_fine = compute_shear_tensor(vel_x, vel_y, vel_z, box_size=box_size, H0=67.5)
    
    print("Diagonalizing fine shear tensor...")
    lambdas_fine, evecs = diagonalize_shear_tensor(sigma_fine)
    del sigma_fine
    gc.collect()
    print("Classifying fine cosmic web...")
    web_fine = classify_cosmic_web(lambdas_fine, lam_th=threshold)
    del lambdas_fine, evecs
    gc.collect()
    # Compute coarse web for multiscale correction


    print("Computing coarse V-web for multiscale correction...")
    vel_x_coarse = gaussian_filter(vel_x, sigma=4, mode='wrap')
    vel_y_coarse = gaussian_filter(vel_y, sigma=4, mode='wrap') 
    vel_z_coarse = gaussian_filter(vel_z, sigma=4, mode='wrap')

    del vel_x, vel_y, vel_z
    gc.collect()
    
    print("Computing coarse shear tensor...")
    sigma_coarse = compute_shear_tensor(vel_x_coarse, vel_y_coarse, vel_z_coarse, box_size=box_size, H0=67.5)
    del vel_x_coarse, vel_y_coarse, vel_z_coarse
    gc.collect()
    print("Diagonalizing coarse shear tensor...")
    lambdas_coarse, _ = diagonalize_shear_tensor(sigma_coarse)
    del sigma_coarse
    gc.collect()
    print("Classifying coarse cosmic web...")
    web_coarse = classify_cosmic_web(lambdas_coarse, lam_th=threshold)
    del lambdas_coarse
    gc.collect()
        
    # Apply multiscale correction
    print("Applying multiscale correction...")
    web = apply_multiscale_correction(web_fine, web_coarse, density_grid, mean_density=1.0, virial_density=340.0)

    del web_coarse, web_fine, density_grid
    gc.collect()

    plotting_routine(web,box_size,grid_size,threshold)
    #Add whatever plots you like here, a lot of the data is deleted to conserve RAM.
    #Remove the del for the data you want or re-read it in.

    print("Finished :).")


def build_velocity_grid(positions: np.ndarray,
                        velocities: np.ndarray,
                        box_size: float,
                        grid_size: int = 100,
                        method: str = "ngp",
                        vel_x: np.ndarray = None,
                        vel_y: np.ndarray = None,
                        vel_z: np.ndarray = None,
                        count: np.ndarray = None,
                        average: bool = False):
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

    #Average velocities in each cell.
    if average:
        mask = count > 0
        vel_x[mask] /= count[mask]
        vel_y[mask] /= count[mask]
        vel_z[mask] /= count[mask]
        vel_x[~mask] = 0.0
        vel_y[~mask] = 0.0
        vel_z[~mask] = 0.0

    return vel_x, vel_y, vel_z, count

def build_mass_grid(positions: np.ndarray,
                    masses: np.ndarray,
                    box_size: float,
                    grid_size: int = 100,
                    method: str = "ngp"
                    mass_grid:np.ndarray,):
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
    mass_grid = gaussian_filter(mass_grid, sigma=1, mode="wrap")
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
    vecs_sorted = np.take_along_axis(vecs, idx[:, None, :], axis=2)

    lambda1 = vals_sorted[:, 0].reshape(N, N, N)
    lambda2 = vals_sorted[:, 1].reshape(N, N, N)
    lambda3 = vals_sorted[:, 2].reshape(N, N, N)
    evec1 = vecs_sorted[:, :, 0].reshape(N, N, N, 3)
    evec2 = vecs_sorted[:, :, 1].reshape(N, N, N, 3)
    evec3 = vecs_sorted[:, :, 2].reshape(N, N, N, 3)

    return (lambda1, lambda2, lambda3), (evec1, evec2, evec3)

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
    

    plt.savefig(f"MULTI_Cosmic_web_multiscale_no_part_gridsize_{grid_size}_{threshold}.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    cosmic_web_classification_routine()




