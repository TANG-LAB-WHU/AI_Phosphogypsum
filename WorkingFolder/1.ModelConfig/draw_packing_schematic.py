
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_schematic(output_file="packing_schematic.png"):
    fig, ax = plt.figure(figsize=(10, 8)), plt.gca()
    
    # Parameters for visualization
    slab_height = 10.0
    z_buffer = 2.0
    packing_height = 15.0
    target_vacuum = 5.0  # Optional extra vacuum
    
    x_width = 10.0
    
    # Coordinates
    z0 = 0
    z1 = z0 + slab_height
    z2 = z1 + z_buffer
    z3 = z2 + packing_height
    z4 = z3 + target_vacuum 
    
    # 1. Draw Slab
    slab_rect = patches.Rectangle((0, z0), x_width, slab_height, linewidth=1, edgecolor='none', facecolor='#A9A9A9', alpha=0.5)
    ax.add_patch(slab_rect)
    ax.text(x_width/2, z0 + slab_height/2, 'Slab Region\n(CaSO4)', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # 2. Draw z_buffer region
    buffer_rect = patches.Rectangle((0, z1), x_width, z_buffer, hatch='//', edgecolor='#90EE90', facecolor='none', alpha=0.5)
    ax.add_patch(buffer_rect)
    
    # 3. Draw Packing region
    pack_rect = patches.Rectangle((0, z2), x_width, packing_height, linewidth=2, edgecolor='#1E90FF', facecolor='#E6F3FF', alpha=0.3)
    ax.add_patch(pack_rect)
    ax.text(x_width/2, z2 + packing_height/2, 'Packing Region\n(Molecules added here)', ha='center', va='center', fontsize=12, color='#00008B')

    # Draw dummy molecules
    import numpy as np
    np.random.seed(42)
    for _ in range(15):
        mx = np.random.uniform(1, x_width-1)
        mz = np.random.uniform(z2 + 0.5, z3 - 0.5)
        circle = patches.Circle((mx, mz), 0.3, edgecolor='black', facecolor='red', alpha=0.7)
        ax.add_patch(circle)

    # 4. Draw Vacuum above (if any)
    # Just a line for the top of the cell
    
    # Annotations (Arrows)
    
    # Slab Height
    ax.annotate('', xy=(-1, z0), xytext=(-1, z1), arrowprops=dict(arrowstyle='<->', lw=1.5))
    ax.text(-1.5, (z0+z1)/2, 'Slab\nThickness', ha='center', va='center', rotation=90)
    
    # z_buffer
    ax.annotate('', xy=(-1, z1), xytext=(-1, z2), arrowprops=dict(arrowstyle='<->', lw=1.5, color='green'))
    ax.text(-1.5, (z1+z2)/2, 'z_buffer', ha='center', va='center', rotation=90, color='green')
    
    # packing_vacuum
    ax.annotate('', xy=(-1, z2), xytext=(-1, z3), arrowprops=dict(arrowstyle='<->', lw=1.5, color='blue'))
    ax.text(-1.5, (z2+z3)/2, 'packing_vacuum\n(Height)', ha='center', va='center', rotation=90, color='blue')
    
    # Top of Cell
    ax.axhline(y=z3, color='black', linestyle='--', linewidth=2)
    ax.text(x_width + 0.5, z3, 'Top of Cell (if no target_vacuum)', va='center')

    # Target Vacuum
    # If target_vacuum is set, the cell top moves
    ax.annotate('', xy=(x_width + 2, z3), xytext=(x_width + 2, z3 + 5), arrowprops=dict(arrowstyle='->', linestyle='dashed'))
    ax.text(x_width + 2.5, z3 + 2.5, 'Extended if\ntarget_vacuum > current', va='center')
    
    # Z-axis lines
    ax.axhline(y=z1, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=z2, color='green', linestyle='--', alpha=0.5)
    
    # Limits and Labels
    ax.set_xlim(-3, x_width + 5)
    ax.set_ylim(-1, z4 + 2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.title('Molecule Packing Vertical Layout', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Schematic saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="packing_schematic.png", help="Output filename")
    args = parser.parse_args()
    
    draw_schematic(args.output)
