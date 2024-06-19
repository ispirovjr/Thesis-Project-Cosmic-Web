import numpy as np
import sys

class DensityHeader:
    def __init__(self, gridSize=None, totalGrid=None, fileType=None, noDensityFiles=None, densityFileGrid=None, indexDensityFile=None, box=None, npartTotal=None, mass=None, time=None, redshift=None, BoxSize=None, Omega0=None, OmegaLambda=None, HubbleParam=None, method=None, fill=None, FILE_ID=None):
        self.gridSize = gridSize
        self.totalGrid = totalGrid
        self.fileType = fileType
        self.noDensityFiles = noDensityFiles
        self.densityFileGrid = densityFileGrid
        self.indexDensityFile = indexDensityFile
        self.box = box
        self.npartTotal = npartTotal
        self.mass = mass
        self.time = time
        self.redshift = redshift
        self.BoxSize = BoxSize
        self.Omega0 = Omega0
        self.OmegaLambda = OmegaLambda
        self.HubbleParam = HubbleParam
        self.method = method 
        self. fill = fill
        self.FILE_ID = FILE_ID


def write_file(file_name, header, data):
    with open(file_name, 'wb') as f:
        header_buffer = np.int64(1024)
        header_buffer.tofile(f)

        header.gridSize.tofile(f)
        header.totalGrid.tofile(f)
        header.fileType.tofile(f)
        header.noDensityFiles.tofile(f)
        header.densityFileGrid.tofile(f)
        header.indexDensityFile.tofile(f)
        header.box.tofile(f)

        header.npartTotal.tofile(f)
        header.mass.tofile(f)
        header.time.tofile(f)
        header.redshift.tofile(f)
        header.BoxSize.tofile(f)
        header.Omega0.tofile(f)
        header.OmegaLambda.tofile(f)
        header.HubbleParam.tofile(f)

        header.method.tofile(f)
        header.fill.tofile(f)
        header.FILE_ID.tofile(f)


        header_buffer.tofile(f)


        data_bytes = data.nbytes

        buffer = np.array([data_bytes], dtype=np.uint64)

        buffer.tofile(f)
        data.tofile(f)

        buffer.tofile(f)



def main():
    if len(sys.argv) != 2:
        print("Usage: python convert-dens.py filename.npy")
        sys.exit(1)

    argument = sys.argv[1]
    print(f"Converting file: {argument}")

    # Load the file 
    dens = np.load(str(argument)).astype(np.float32)
    grid = len(dens)
    print(dens.shape)
# Populate the header with some example values
    header = DensityHeader()
    header.gridSize = np.array([grid, grid, grid], dtype=np.uint64)  # Example grid size
    header.totalGrid = np.uint64(header.gridSize.prod())  # Total number of grid points
    header.fileType = np.int32(1)  # Example file type
    header.noDensityFiles = np.int32(1)  # Assuming single file for simplicity
    header.densityFileGrid = np.array([1, 1, 1], dtype=np.int32)
    header.indexDensityFile = np.int32(0)
    header.box = np.array([0, grid, 0, grid, 0, grid], dtype=np.float64)  # Example box coordinates

# Gadget snapshot related data
    header.npartTotal = np.array([0, grid, 0, 0, 0, 0], dtype=np.uint64) #np.zeros(6, dtype=np.uint64)
    header.mass = np.array([0.0, 0.03388571, 0.0, 0.0, 0.0, 0.0], dtype=np.float64) #np.zeros(6, dtype=np.float64)
    header.time = np.float64(0.9999999999999998)
    header.redshift = np.float64(0.0)
    header.BoxSize = np.float64(128.0)
    header.Omega0 = np.float64(0.2726)
    header.OmegaLambda = np.float64(0.7274)
    header.HubbleParam = np.float64(0.704)

    header.method = np.uint64(1)
    header.fill = np.zeros(760, dtype='c')
    header.FILE_ID = np.int64(1)

# Set types correctly
# header.SetType()

# Create example data
    data = dens
# File name to write
    file_name = argument[:-4] + ".dat"
    write_file(file_name, header, data)

# Write header and data to the file

if __name__ == "__main__":
    main()
