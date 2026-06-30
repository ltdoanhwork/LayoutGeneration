"""
Coordinate transformation utilities for collage assembly.
Handle conversion between different coordinate systems:
- World coordinates (x, y) where y increases upward
- Canvas coordinates (row, col) where row increases downward
"""

import numpy as np


def rowcol2xy(row, col, ymax):
    """
    Convert row/column coordinates to x/y world coordinates.
    
    Args:
        row: Row index (0 at top, increases downward)
        col: Column index
        ymax: Maximum y value (canvas height)
    
    Returns:
        (x, y) tuple where y=0 at bottom
    """
    return int(col), int(ymax - row)


def xy2rowcol(x, y, ymax):
    """
    Convert x/y world coordinates to row/column canvas coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate (0 at bottom, increases upward)
        ymax: Maximum y value (canvas height)
    
    Returns:
        (row, col) tuple where row=0 at top
    """
    return int(round(ymax - y, 0)), int(round(x, 0))


def polygon2local_coordinate(polygon):
    """
    Move polygon origin to (minX, minY) of bounding box.
    
    Args:
        polygon: Shapely polygon
    
    Returns:
        numpy array of local coordinates
    """
    bounding_box = polygon.bounds
    return np.array([
        (int(coord[0] - bounding_box[0]), int(coord[1] - bounding_box[1]))
        for coord in list(polygon.exterior.coords)
    ])


def get_shape_centroid_ratio(polygon):
    """
    Get the centroid position as ratio within bounding box.
    
    Args:
        polygon: Shapely polygon
    
    Returns:
        (rx, ry) where centroid is at rx * width, ry * height from top-left
    """
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    
    centroid = polygon.centroid
    rx = (centroid.x - bounds[0]) / w if w > 0 else 0.5
    ry = (centroid.y - bounds[1]) / h if h > 0 else 0.5
    
    return (rx, ry)


def triangulation(outer_rec, inner_rec, height):
    """
    Get the triangulation given outer and inner rectangles.
    
    Rectangles are in counter-clockwise order starting from (0,0):
        [bottom left, bottom right, top right, top left]
    
    Args:
        outer_rec: Outer rectangle vertices
        inner_rec: Inner rectangle vertices
        height: Image height for coordinate transformation
    
    Returns:
        List of triangles with y-coordinates flipped
    """
    triangles = [
        [outer_rec[3], inner_rec[3], outer_rec[2]],
        [inner_rec[3], inner_rec[2], outer_rec[2]],
        [inner_rec[2], outer_rec[1], outer_rec[2]],
        [inner_rec[2], inner_rec[1], outer_rec[1]],
        [inner_rec[0], outer_rec[1], inner_rec[1]],
        [outer_rec[0], outer_rec[1], inner_rec[0]],
        [outer_rec[0], inner_rec[0], inner_rec[3]],
        [outer_rec[3], outer_rec[0], inner_rec[3]],
        [inner_rec[3], inner_rec[2], inner_rec[1]],
        [inner_rec[0], inner_rec[1], inner_rec[3]]
    ]
    return [[(vertex[0], height - vertex[1]) for vertex in t] for t in triangles]


def world_to_canvas_bbox(bounding_box, canvas_height):
    """
    Convert world coordinates bounding box to canvas row/col coordinates.
    
    Args:
        bounding_box: (minx, miny, maxx, maxy) in world coordinates
        canvas_height: Height of canvas
    
    Returns:
        (row_start, col_start, row_end, col_end) in canvas coordinates
    """
    minx, miny, maxx, maxy = bounding_box
    
    # In world coords: y increases upward, maxy is top
    # In canvas coords: row increases downward, row 0 is top
    row_start = int(round(canvas_height - maxy, 0))
    row_end = int(round(canvas_height - miny, 0))
    col_start = int(round(minx, 0))
    col_end = int(round(maxx, 0))
    
    return (row_start, col_start, row_end, col_end)


def canvas_to_world_point(row, col, canvas_height):
    """
    Convert canvas point to world coordinates.
    
    Args:
        row: Row in canvas (0 at top)
        col: Column in canvas
        canvas_height: Height of canvas
    
    Returns:
        (x, y) in world coordinates (y=0 at bottom)
    """
    x = col
    y = canvas_height - row
    return (x, y)


def world_to_canvas_point(x, y, canvas_height):
    """
    Convert world point to canvas coordinates.
    
    Args:
        x: X in world coordinates
        y: Y in world coordinates (0 at bottom)
        canvas_height: Height of canvas
    
    Returns:
        (row, col) in canvas coordinates (row=0 at top)
    """
    row = int(round(canvas_height - y, 0))
    col = int(round(x, 0))
    return (row, col)
