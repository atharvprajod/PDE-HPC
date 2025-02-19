import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve

def hilbert_key(point, order=3):
    """Convert 3D coordinate to Hilbert index"""
    scaled = [int(p * (2**order - 1)) for p in point]
    hc = HilbertCurve(order, 3)
    return hc.distance_from_coordinates(scaled)

def find_adjacent_partitions(part, all_parts):
    """Find neighboring partitions using cell adjacency"""
    neighbors = set()
    for cell in part.cells:
        for adj in cell.adjacent:
            for p in all_parts:
                if adj in p.cells and p != part:
                    neighbors.add(p.id)
    return list(neighbors)

def decompose_domain(mesh, n_procs):
    coords = [cell.center for cell in mesh.cells]
    keys = {i: hilbert_key(c) for i, c in enumerate(coords)}
    sorted_ids = sorted(keys, key=lambda x: keys[x])
    
    partitions = np.array_split(sorted_ids, n_procs)
    return [
        MeshPartition(
            cells=part,
            neighbors=find_adjacent_partitions(...),
            ghost_layers=1
        ) for part in partitions
    ]