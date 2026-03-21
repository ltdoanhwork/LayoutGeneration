import sys
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx 

# Add parent directory to path to find shape_decomposition and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shape_decomposition as sd
from utils.get_mask import predict_mask, preprocess_image, refine_mask


def _plot_polygon(ax, polygon, face_color="#d0d0d0", edge_color="#444444"):
    if polygon is None or polygon.is_empty:
        return
    exterior = np.array(polygon.exterior.coords)
    ax.fill(exterior[:, 0], exterior[:, 1], facecolor=face_color, edgecolor=edge_color, linewidth=2)
    for interior in polygon.interiors:
        hole = np.array(interior.coords)
        ax.fill(hole[:, 0], hole[:, 1], facecolor="white", edgecolor=edge_color, linewidth=1)


def _plot_multilinestring(ax, mls, color="#ff6a00", linewidth=2.5):
    if mls is None:
        return
    if mls.geom_type == "MultiLineString":
        for line in mls.geoms:
            pts = np.array(line.coords)
            ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth, 
                   solid_capstyle='round', solid_joinstyle='round', alpha=0.9)
    elif mls.geom_type == "LineString":
        pts = np.array(mls.coords)
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth,
               solid_capstyle='round', solid_joinstyle='round', alpha=0.9)


def _plot_corners(ax, corner_dict, color="#ff3b30"):
    for key in corner_dict:
        if isinstance(corner_dict[key], sd.Corner):
            ax.plot([key[0]], [key[1]], marker="o", markersize=5, color=color)


def _plot_cuts(ax, cuts, color="#ffb020", alpha=0.8, linewidth=2):
    for cut in cuts:
        ax.plot([cut[0][0], cut[1][0]], [cut[0][1], cut[1][1]],
                color=color, alpha=alpha, linewidth=linewidth)

def plot_skeleton_overlay(G, distance_map, output_path, title="Skeleton Overlay"):
    """
    Plots the graph skeleton overlay on the distance map.
    Handles coordinate conversion between Shape Decomposition (y-up) and Image (y-down).
    """
    h, w = distance_map.shape
    plt.figure(figsize=(12, 12))
    
    # Display distance map with origin='upper' (standard image view, 0,0 at top-left)
    plt.imshow(distance_map, cmap='gray', origin='upper')
    
    # SD coordinates: y=0 is BOTTOM.
    # To plot on origin='upper' (y=0 is TOP):
    # plot_y = h - y
    
    # Plot Edges
    for u, v in G.edges():
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        
        py1 = h - y1
        py2 = h - y2
        
        plt.plot([x1, x2], [py1, py2], 'r-', linewidth=1.5, alpha=0.7)
        
    # Plot Nodes
    for n in G.nodes():
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        py = h - y
        plt.plot(x, py, 'b.', markersize=4, alpha=0.8)

    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def debug_medial_axis(input_shape, output_dir):
    print(f"Processing shape: {input_shape}")
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_shape)
    if image is None:
        print(f"Error: Cannot load image {input_shape}")
        return

    # Preprocess (Mask)
    print("Generating mask...")
    from utils.get_mask import net
    inputs, orig_h, orig_w = preprocess_image(image)
    pred_mask = predict_mask(net, inputs)
    mask_refined = refine_mask(pred_mask, orig_h, orig_w)
    
    cv2.imwrite(os.path.join(output_dir, "debug_mask.png"), mask_refined)

    # Medial Axis Input (Shape=True)
    binary_input = mask_refined < 127
    
    print("Calculating Ridges (Medial Axis)...")
    # Ridge Thresholds (can adjust here if needed)
    # Default in SD: ridge_threshold=0.3, small_threshold=5
    medial_axis_group, distance_map = sd.ridge_medial_axis(binary_input, ridge_threshold=0.3, small_threshold=5)
    
    # Save raw medial axis
    plt.figure(figsize=(10,10))
    plt.imshow(medial_axis_group, cmap='jet')
    plt.title("Raw Medial Axis Groups")
    plt.savefig(os.path.join(output_dir, "debug_raw_medial_axis.png"))
    plt.close()

    print("Building Graph Skeleton...")
    try:
        # Build multilinestring
        multilinestring, line_labels = sd.build_medial_multilinestring(medial_axis_group)
        
        # Redistributions & Graph
        if multilinestring.is_empty:
             print("Warning: Medial axis multilinestring is empty!")
             return

        final_medial_vertices_int = sd.redistribute_vertices(multilinestring, 5)
        G_int = sd.build_medial_graph(final_medial_vertices_int, line_labels, distance_map)
        
        # Plot Skeleton
        output_skel = os.path.join(output_dir, "debug_skeleton_viz.png")
        plot_skeleton_overlay(G_int, distance_map, output_skel)
        print(f"Saved skeleton debug to {output_skel}")

        # Clean medial axis layout visualization
        try:
            poly = sd.generate_canvas_polygon(mask_refined)[0]
            end_vertices = sd.find_end_vertices(G_int, exterior=False)

            fig, ax = plt.subplots(figsize=(8, 8))
            _plot_polygon(ax, poly, face_color="#d0d0d0", edge_color="#b0b0b0")
            _plot_multilinestring(ax, multilinestring, color="#ff6a00", linewidth=3.0)

            # End vertices (medial axis endpoints)
            for v in end_vertices:
                x, y = G_int.nodes[v]['x'], G_int.nodes[v]['y']
                ax.plot([x], [y], marker="o", markersize=6, color="#ff3b30")

            ax.set_title("Medial axis layout")
            ax.set_aspect('equal')
            ax.axis("off")
            out_layout = os.path.join(output_dir, "medial_axis_layout.png")
            fig.savefig(out_layout, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Failed to create medial_axis_layout.png: {e}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to build/plot graph: {e}")


def debug_medial_axis_steps(input_shape, output_dir):
    print(f"[DEBUG] Medial axis step-by-step for: {input_shape}")
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(input_shape, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Cannot load image {input_shape}")
        return

    polygon = sd.generate_canvas_polygon(image)[0]
    medial_exterior_input = sd.prepare_for_medial_axis(image, complement=True)
    medial_interior_input = sd.prepare_for_medial_axis(image, complement=False)

    ma_ext = sd.ridge_medial_axis(medial_exterior_input, ridge_threshold=0.3, small_threshold=5)
    ma_int = sd.ridge_medial_axis(medial_interior_input, ridge_threshold=0.3, small_threshold=5)

    if ma_ext[0].sum() <= 0:
        print("[DEBUG] No concave corners detected; step-by-step skipped.")
        return

    # Build medial axes
    multilinestring_ext = sd.build_medial_multilinestring(ma_ext[0])
    multilinestring_int = sd.build_medial_multilinestring(ma_int[0])
    final_medial_vertices_ext = sd.redistribute_vertices(multilinestring_ext[0], 5)
    final_medial_vertices_int = sd.redistribute_vertices(multilinestring_int[0], 5)
    G_ext = sd.build_medial_graph(final_medial_vertices_ext, multilinestring_ext[1], ma_ext[1])
    G_int = sd.build_medial_graph(final_medial_vertices_int, multilinestring_int[1], ma_int[1])
    endv_ext = sd.find_end_vertices(G_ext, exterior=True)
    endv_int = sd.find_end_vertices(G_int, exterior=False)

    boundary_vertices = sd.redistribute_vertices(sd.LineString(polygon.exterior.coords), 5)
    boundary_vertices_dict_ext = sd.build_boundary_dic(boundary_vertices)
    corner_mapping_ext = sd.mark_corners(G_ext, endv_ext, boundary_vertices_dict_ext, boundary_vertices)
    component_adjusted_ext = sd.mark_component(boundary_vertices_dict_ext)

    boundary_vertices_dict_int = sd.build_boundary_dic(boundary_vertices)
    corner_mapping_int = sd.mark_corners(G_int, endv_int, boundary_vertices_dict_int, boundary_vertices)
    component_adjusted_int = sd.mark_component(boundary_vertices_dict_int)

    ec = sd.mark_extended_corner(component_adjusted_ext, component_adjusted_int, boundary_vertices, G_ext, endv_ext, corner_mapping_ext)
    ec_adjusted = sd.adjust_corner(ec)

    projection_pairs = sd.extract_projection_pair(G_int, boundary_vertices)
    raw_cuts = sd.generate_raw_cuts(projection_pairs, component_adjusted_ext)
    representative = sd.select_representative_cuts(raw_cuts, component_adjusted_ext, component_adjusted_int)
    denoised = [r for r in representative if sd.protrusion_strength(r[0], r[1], boundary_vertices) < 0.75]

    # Ranking
    priority = np.array([1 if sd.isDouble(r, component_adjusted_ext) else 0 for r in denoised])
    eal = sd.get_extented_arc_length(ec_adjusted, distance=5)
    protrusion_threshold = 0.5
    extension_strength_threshold = 0.9
    protrusion_strength_filter = np.array([sd.protrusion_strength(r[0], r[1], boundary_vertices) > protrusion_threshold for r in denoised])
    extension_strength_filter = np.array([sd.extension_strength(r, eal, component_adjusted_ext) < extension_strength_threshold for r in denoised])
    priority_decrease = np.logical_or(protrusion_strength_filter, extension_strength_filter).astype(int)
    saliency_adjusted_priority = priority - priority_decrease

    corner_source, corner_endpoints = sd.extract_corner_info(component_adjusted_ext)
    calibrated = sd.corner_calibrated_endpoints(corner_mapping_ext, corner_endpoints, boundary_vertices, G_ext)
    cut_2_corner, corner_2_cuts = sd.cut_corner_mappings(denoised, component_adjusted_ext)
    _, corner2rawcuts = sd.cut_corner_mappings(raw_cuts, component_adjusted_ext)
    corner_residue = sd.calculate_corner_residue(corner2rawcuts, boundary_vertices)
    corner_ordered = [corner for corner, _ in sorted(corner_residue.items(), key=lambda item: item[1])]

    angle_equivalence_threshold = np.pi / 20.0
    tolerance = np.pi / 12.0
    final_cuts = []
    for corner in corner_ordered:
        if corner in corner_2_cuts:
            cuts = corner_2_cuts[corner].copy()
            cuts.sort(key=lambda x: saliency_adjusted_priority[denoised.index(x)])
            start_vector = sd.point_tangent_vector(boundary_vertices, calibrated[corner][0], is_first=True)
            end_vector = sd.point_tangent_vector(boundary_vertices, calibrated[corner][1], is_first=False)
            concave_corner = sd.InteriorAngle(start_vector, end_vector)
            concave_corner.tolerance = tolerance

            for cut in cuts:
                if cut in final_cuts:
                    cut_vector = sd.create_cut_vector(cut, corner, component_adjusted_ext)
                    concave_corner.add_cut(cut_vector)

            if concave_corner.is_convex():
                continue

            while cuts:
                current_cut = cuts.pop()
                if current_cut in final_cuts:
                    continue
                cut_vector = sd.create_cut_vector(current_cut, corner, component_adjusted_ext)
                difference = [sd.interior_angle(cut_vector, past_cut_angle) for past_cut_angle in concave_corner.cut_list]
                if all(d > angle_equivalence_threshold for d in difference):
                    concave_corner.add_cut(cut_vector)
                    if current_cut not in final_cuts:
                        final_cuts.append(current_cut)
                if concave_corner.is_convex():
                    break

    # Step 1: Construct medial axes
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_polygon(ax, polygon)
    _plot_multilinestring(ax, multilinestring_int[0], color="#ff6a00", linewidth=3.0)
    ax.set_title("Step 1 - Construct medial axis")
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, "step_1_medial_axis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Step 2: Find concave corners
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_polygon(ax, polygon)
    _plot_corners(ax, component_adjusted_ext, color="#ff3b30")
    ax.set_title("Step 2 - Find concave corners")
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, "step_2_concave_corners.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Step 3: Generate raw cuts
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_polygon(ax, polygon)
    _plot_cuts(ax, raw_cuts, color="#ff9500", alpha=0.7, linewidth=2)
    ax.set_title("Step 3 - Generate raw cuts")
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, "step_3_raw_cuts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Step 4: Rank cuts by saliency
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_polygon(ax, polygon)
    for i, cut in enumerate(denoised):
        if saliency_adjusted_priority[i] >= 1:
            _plot_cuts(ax, [cut], color="#ff3b30", alpha=0.85, linewidth=2.5)
        else:
            _plot_cuts(ax, [cut], color="#4cd964", alpha=0.7, linewidth=2)
    ax.set_title("Step 4 - Rank cuts by saliency")
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, "step_4_ranked_cuts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Step 5: Select cuts until convex
    fig, ax = plt.subplots(figsize=(8, 8))
    _plot_polygon(ax, polygon)
    _plot_cuts(ax, final_cuts, color="#ff3b30", alpha=0.9, linewidth=2.5)
    ax.set_title("Step 5 - Selected cuts")
    ax.axis("off")
    fig.savefig(os.path.join(output_dir, "step_5_selected_cuts.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_medial_axis.py <input_shape> <output_dir> ...")
        sys.exit(1)
        
    input_shape = sys.argv[1]
    
    # Try to find output dir in 3rd arg (run.py format)
    # python script img folder output ...
    output_dir = "output_debug_medial_axis"
    # Logic to ignore flags and find 3rd positional arg
    args = [a for a in sys.argv if not a.startswith('--')]
    if len(args) >= 4:
        # args[0] is script, args[1] is img, args[2] is folder, args[3] is output
        output_dir = args[3]
    elif len(args) == 3:
        output_dir = args[2]
         
    debug_medial_axis(input_shape, output_dir)
    debug_medial_axis_steps(input_shape, output_dir)
