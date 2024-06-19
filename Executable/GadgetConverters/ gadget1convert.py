import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import numpy.fft
import numba
from numba import float32, float64, int64
from typing import Union
import io
from scipy.spatial import Delaunay
from scipy import ndimage as ndi

# Function to import particle positions, dw about it, it's only so you have a reference for the variable names
def importFiles(timestep, path='./z0/'):
    c_coordinates = None
    c_velocities = None
    c_ids = None
    for file in os.listdir(path):
        if 'hdf5' in file:
            if timestep in file:
                coordinates = h5py.File(path + file, 'r')['PartType1']['Coordinates'][:]
                velocities = h5py.File(path + file, 'r')['PartType1']['Velocities'][:]
                ids = h5py.File(path + file, 'r')['PartType1']['ParticleIDs'][:]
                if c_coordinates is None:
                    c_coordinates = coordinates
                if c_velocities is None:
                    c_velocities = velocities
                if c_ids is None:
                    c_ids = ids
                else:
                    c_coordinates = np.concatenate((c_coordinates, coordinates), axis = 0)
                    c_velocities = np.concatenate((c_velocities, velocities), axis = 0)
                    c_ids = np.concatenate((c_ids, ids), axis = 0)

    c_coordinates = c_coordinates/1000 # changing the scale to Mpc/h
    c_coordiantes = c_coordinates.astype(np.float32)

    return c_coordinates, c_velocities, c_ids

# Write the Gadget-1 file with Fortran-style record markers
def write_gadget(ds_coordinates, ds_velocities, ds_ids):

    def write_fortran_record(f, data):
        record_size = len(data)
        f.write(struct.pack('i', record_size))
        f.write(data)
        f.write(struct.pack('i', record_size))

    with open('gadget_snap_z0.0_rand_red.dat', 'wb') as f:
        # Write the header
        write_fortran_record(f, header.tobytes())

        # Write the particle positions
        write_fortran_record(f, ds_coordinates.tobytes())

        # Write the particle velocities
        write_fortran_record(f, ds_velocities.tobytes())

        # Write the particle IDs
        write_fortran_record(f, ds_ids.tobytes())

    print("Gadget-1 file 'gadget_snap_z0.0_rand_red.dat' created with Fortran-style record markers.")

header_dtype = np.dtype([
    ('npart', (np.int32, 6)),         # Number of particles of each type
    ('mass', (np.float64, 6)),        # Mass of each particle type
    ('time', np.float64),             # Time of the snapshot
    ('redshift', np.float64),         # Redshift of the snapshot
    ('flag_sfr', np.int32),           # Star formation flag
    ('flag_feedback', np.int32),      # Feedback flag
    ('npartTotal', (np.int32, 6)),    # Total number of particles of each type
    ('flag_cooling', np.int32),       # Cooling flag
    ('num_files', np.int32),          # Number of files in multi-file set
    ('BoxSize', np.float64),          # Box size of the simulation
    ('Omega0', np.float64),           # Matter density parameter
    ('OmegaLambda', np.float64),      # Cosmological constant density parameter
    ('HubbleParam', np.float64),      # Hubble parameter
    ('flag_stellarage', np.int32),    # Stellar age flag
    ('flag_metals', np.int32),        # Metals flag
    ('npartTotalHighWord', (np.int32, 6)),  # High word of total number of particles
    ('flag_entropy_instead_u', np.int32),   # Entropy flag
    ('flag_doubleprecision', np.int32),     # Double precision flag
    ('flag_ic_info', np.int32),             # IC info flag
    ('lpt_scalingfactor', np.float32),      # LPT scaling factor
    ('fill', (np.int32, 12))          # Padding to make header 256 bytes
])

# Initialize the header with example values
header = np.zeros(1, dtype=header_dtype)
header['npart'] = [0,len(rand_pos_red_norm), 0, 0, 0, 0] # Replace with whatever number of particles/subhalos you have (has to be the second element, it specifies it's DM)
header['mass'] = [0.0, 0.03388571, 0.0, 0.0, 0.0, 0.0]
header['time'] = 0.
header['redshift'] = 0.
header['flag_sfr'] = 0
header['flag_feedback'] = 0
header['npartTotal'] = [0, len(rand_pos_red_norm), 0, 0, 0, 0] #was 94196375
header['flag_cooling'] = 0
header['num_files'] = 1 #was 8
header['BoxSize'] = 75.0  # Ensuring it's a large enough box
header['Omega0'] = 0.2726
header['OmegaLambda'] = 0.7274
header['HubbleParam'] = 0.704
header['flag_stellarage'] = 0
header['flag_metals'] = 0
header['npartTotalHighWord'] = [0, 0, 0, 0, 0, 0]
header['flag_entropy_instead_u'] = 0
header['flag_doubleprecision'] = 0
header['flag_ic_info'] = 0
header['lpt_scalingfactor'] = 1.0
header['fill'] = [0] * 12

write_gadget(rand_pos_red_norm, rand_vel, ids) # ofc replace with whatever you need here