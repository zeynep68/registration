"""
Parallel implementation of marching cubes algorithm.

Adapted from https://jugit.fz-juelich.de/c.schiffer/bigbrain_reconstruction/-/blob/master/bigbrain_reconstruction/cli/marching_cubes.py
"""

def marching_cubes(volume, output_fname, volume_prefix, affine_prefix, chunk_size, level, pad_chunks=False):
    """
    Takes an HDF5 volume file and creates an GII mesh file out of it.
    Can be run in parallel to process extremely large volumes.
    """
    import h5py
    import numpy as np
    from atlasmpi import MPI
    from meshio import write_mesh
    from typhon.algorithms.apply import apply_to_chunks

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    surface_collector = SurfaceCollector()

    # Read the volume
    print(f"Opening volume at {volume}...")
    with h5py.File(volume, "r") as f:
        volume_dataset = f[volume_prefix]
        affine_dataset = f[affine_prefix]
        volume_shape = volume_dataset.shape
        volume_dtype = volume_dataset.dtype
        print(f"Volume shape: {volume_shape} (dtype: {volume_dtype})")

        print("Reading affine matrix...")
        affine = affine_dataset[...]
        spacing = (affine[0, 0], affine[1, 1], affine[2, 2])
        print(f"Spacing [mm]: {spacing}")
        origin = affine[:3, -1]
        print(f"Origin [mm]: {origin}")
        # Number of chunks per dimension
        num_chunks = np.ceil(np.array(volume_dataset.shape) / chunk_size)

        print("Applying marching cubes to chunks")
        apply_to_chunks(array=volume_dataset,
                        function=_marching_cubes_on_chunk,
                        ghost_points=1,
                        chunk_size=chunk_size,
                        kwargs={
                            "surface_collector": surface_collector,
                            "chunk_size": chunk_size,
                            "level": level,
                            "num_chunks": num_chunks,
                            "pad_chunks": pad_chunks,
                        },
                        write_back=False,
                        rank=rank,
                        size=size,
                        pass_valid_pixel_slice=True,
                        log_interval=10,
                        log_fn=print,
                        log_prefix="chunk_surface")

    print("Finished running marching cubes to chunks")

    # Shift meshes to correct physical origin
    print(f"Shifting vertices to spatial origin and adjusting spacing")
    vertices = surface_collector.chunk_vertices
    faces = surface_collector.chunk_faces

    if vertices:
        for verts in vertices:
            verts *= spacing
            verts += origin

        # Shift face indices to unify face indices within local processor
        print("Unifying face indices for local vertices")
        index_offsets = np.cumsum([len(f) for f in vertices])
        index_offsets = np.roll(index_offsets, shift=1, axis=0)
        index_offsets[0] = 0
        assert index_offsets.shape[0] == len(faces)
        for shift, face in zip(index_offsets, faces):
            face += shift

        # First, do a local concatenate of the data
        local_vertices = np.concatenate(surface_collector.chunk_vertices, axis=0)
        local_faces = np.concatenate(surface_collector.chunk_faces, axis=0)
    else:
        # Create dummy arrays for processors which did not find any vertices (or got no chunks)
        local_vertices = np.empty(0, dtype=np.float32)
        local_faces = np.empty(0, dtype=np.int32)

    print("Collecting vertices and faces from processors")
    print(f"Data of this processor: {len(local_vertices)} vertices, {len(local_faces)} faces")
    # Distribute the number of elements to expect from each processor
    counts_vertices = np.array(comm.allgather(local_vertices.shape[0]))
    counts_faces = np.array(comm.allgather(local_faces.shape[0]))
    total_vertices = counts_vertices.sum()
    total_faces = counts_faces.sum()

    print("Unifying face indices for global vertices")
    index_offsets = np.cumsum(counts_vertices)
    index_offsets = np.roll(index_offsets, shift=1, axis=0)
    index_offsets[0] = 0
    local_faces += index_offsets[rank]

    # Allocate buffer for global vertices on root
    global_vertices = None
    global_faces = None
    vertices_recv_buf = None
    faces_recv_buf = None
    if rank == 0:
        # Create buffers for vertices and faces
        global_vertices = np.empty((total_vertices, 3), dtype=np.float32)
        global_faces = np.empty((total_faces, 3), dtype=np.int32)
        # Compute the number of elements to receive from each processor
        # Multiply by 3, because each vertex/face has three elements
        vertices_recv_counts = [c * 3 for c in counts_vertices]
        faces_recv_counts = [c * 3 for c in counts_faces]
        # Create the buffers for Gahterv
        vertices_recv_buf = (global_vertices, vertices_recv_counts)
        faces_recv_buf = (global_faces, faces_recv_counts)

    # Gather data on root
    comm.Gatherv(sendbuf=local_vertices, recvbuf=vertices_recv_buf, root=0)
    comm.Gatherv(sendbuf=local_faces, recvbuf=faces_recv_buf, root=0)
    print(f"Collected data from all processors")

    # Write data to file
    if rank == 0:
        print(f"Total mesh size: {len(global_vertices)} vertices, {len(global_faces)} faces")
        print(f"Removing duplicate vertices and faces...")
        combined_vertices, combined_faces = _remove_duplicates(vertices=global_vertices,
                                                               faces=global_faces)
        print(f"Size after reduction: {len(combined_vertices)} vertices, {len(combined_faces)} faces")

        print(f"Writing mesh to file at {output_fname}")
        write_mesh(fname=output_fname,
                   vertices=combined_vertices,
                   faces=combined_faces)

    print("Finished processing, waiting for other processors to finish")
    comm.barrier()


class SurfaceCollector(object):
    """
    Class to collect chunks and vertices.
    """

    def __init__(self):
        self.chunk_vertices = []
        self.chunk_faces = []


def _marching_cubes_on_chunk(chunk, indices, surface_collector, chunk_size, valid_pixel_slice, level, num_chunks, pad_chunks=False):
    """
    Apply marching cubes on a chunk.

    Args:
        chunk (numpy.ndarray): Chunk with ghost points.
        indices (tuple): Indices of the chunk, important to shift indices correctly.
        surface_collector (SurfaceCollector): Helper class to collect results.
        chunk_size (int): Size of the chunk.
        valid_pixel_slice (tuple of slice): Slices for each dimension to determine which points are valid (non-ghost points).
        level (float): Level to extract from the chunk.
        num_chunks (tuple): Number of chunks per dimension.
    """
    import numpy as np
    from skimage.measure import marching_cubes

    # marching_cubes will raise a RunTimeError if
    # no vertices are found, so we check beforehand if there
    # are any foreground pixels
    if chunk.max() == 0:
        return
    # If the minimum is 1, we are completely inside the volume,
    # so no need to surface extraction
    if chunk.min() == 1:
        return
    # Pad border chunks to get closed surfaces
    if pad_chunks:
        pad = []
        for dim, _ in enumerate(chunk.shape):
            p0, p1 = 0, 0
            index = indices[dim]
            chunks_in_dim = num_chunks[dim]
            if index == 0:
                p0 = 1
            if index == chunks_in_dim - 1:
                p1 = 1
            pad.append((p0, p1))
        pad = np.array(pad)
        pad_offset = pad[:, 0]
        # Pad the chunk
        if np.any(pad > 0):
            chunk = np.pad(chunk, pad_width=pad, mode="constant")
    else:
        pad_offset = 0

    # Apply marching cubes to chunk
    zero = np.array(0, dtype=chunk.dtype)
    one = np.array(1, dtype=chunk.dtype)
    chunk = np.where(chunk == level, one, zero)

    try:
        chunk_vertices, chunk_faces, _, _ = marching_cubes(chunk)
    except RuntimeError:
        # Marching cubes throws a RuntimeError if the level does not exist
        return
    # Shift chunk vertices according to the origin of the chunk
    # We need to take care of ghost points by shifting the given
    # start indices of the chunk by the start of the valid pixel slice
    valid_start_indices = np.array([s.start for s in valid_pixel_slice])
    chunk_origin = np.array(indices) * chunk_size - valid_start_indices
    chunk_vertices += chunk_origin - pad_offset
    # Add to collector
    surface_collector.chunk_vertices.append(chunk_vertices)
    surface_collector.chunk_faces.append(chunk_faces)


def _remove_duplicates(vertices, faces):
    """
    Removes duplicate vertices and faces.

    Args:
        vertices (numpy.ndarray): Vertices as array of shape (N, 3).
        faces (numpy.ndarray): Faces as array of shape (M, 3).

    Returns:
        tuple of two numpy.ndarray containing vertices and faces without duplicates.
    """
    import numpy as np

    # Eliminate duplicate vertices
    # We use the inverse to get a mapping which tells us where each element in the original
    # array (vertices) is now located in the new array (new_vertices).
    new_vertices, old_to_new_index_mapping = np.unique(vertices, return_inverse=True, axis=0)

    # Recreate faces array with adjusted vertices by fancy indexing into the index mapping
    # and remove duplicate faces
    new_faces = np.unique(old_to_new_index_mapping[faces], axis=0)

    # Check the results for consistency
    assert new_faces.max() == new_vertices.shape[0] - 1

    return new_vertices, new_faces
