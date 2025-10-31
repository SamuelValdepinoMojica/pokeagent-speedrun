"""
Visual comparison of map sizes for DRL agent observations.
Shows what the agent "sees" with 15x15 vs 7x7 map radius.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_map_coverage():
    """Create visual comparison of map sizes"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # === 15x15 Map (Comprehensive State) ===
    ax1.set_title("Comprehensive State: 15x15 Map\n(225 tiles, ~7 tiles radius)", fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, 14.5)
    ax1.set_ylim(-0.5, 14.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    
    # Draw grid
    for i in range(15):
        for j in range(15):
            # Distance from center
            dist = max(abs(i - 7), abs(j - 7))
            
            if i == 7 and j == 7:
                # Player position
                color = 'red'
                ax1.add_patch(patches.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                                  facecolor=color, edgecolor='black', linewidth=2))
                ax1.text(j, i, 'P', ha='center', va='center', fontsize=12, 
                        fontweight='bold', color='white')
            else:
                # Tiles - gradient based on distance
                alpha = 1.0 - (dist / 7.0) * 0.5
                color = plt.cm.Blues(alpha * 0.8)
                ax1.add_patch(patches.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                                  facecolor=color, edgecolor='gray', linewidth=0.5))
    
    # Add distance markers
    for radius in [3, 5, 7]:
        circle = patches.Circle((7, 7), radius, fill=False, edgecolor='orange', 
                               linewidth=2, linestyle='--', alpha=0.7)
        ax1.add_patch(circle)
        ax1.text(7 + radius + 0.5, 7, f'{radius}', fontsize=10, color='orange', fontweight='bold')
    
    ax1.set_xlabel("X coordinate", fontsize=12)
    ax1.set_ylabel("Y coordinate", fontsize=12)
    
    # === 7x7 Map (Lightweight State) ===
    ax2.set_title("Lightweight State: 7x7 Map\n(49 tiles, 3 tiles radius)", fontsize=14, fontweight='bold')
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-0.5, 6.5)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()
    
    # Draw grid
    for i in range(7):
        for j in range(7):
            # Distance from center
            dist = max(abs(i - 3), abs(j - 3))
            
            if i == 3 and j == 3:
                # Player position
                color = 'red'
                ax2.add_patch(patches.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                                  facecolor=color, edgecolor='black', linewidth=2))
                ax2.text(j, i, 'P', ha='center', va='center', fontsize=12, 
                        fontweight='bold', color='white')
            else:
                # Tiles - gradient based on distance
                alpha = 1.0 - (dist / 3.0) * 0.5
                color = plt.cm.Greens(alpha * 0.8)
                ax2.add_patch(patches.Rectangle((j-0.4, i-0.4), 0.8, 0.8, 
                                                  facecolor=color, edgecolor='gray', linewidth=0.5))
    
    # Add distance marker
    circle = patches.Circle((3, 3), 3, fill=False, edgecolor='green', 
                           linewidth=2, linestyle='--', alpha=0.7)
    ax2.add_patch(circle)
    ax2.text(3 + 3 + 0.5, 3, '3', fontsize=10, color='green', fontweight='bold')
    
    ax2.set_xlabel("X coordinate", fontsize=12)
    ax2.set_ylabel("Y coordinate", fontsize=12)
    
    # Add stats comparison
    stats_text = """
    COMPARISON:
    
    Comprehensive (15x15):
    • Total tiles: 225
    • Vision radius: 7 tiles
    • Memory reads: ~50ms
    • FPS: ~22
    
    Lightweight (7x7):
    • Total tiles: 49 (78% reduction)
    • Vision radius: 3 tiles
    • Memory reads: ~4ms
    • FPS: ~240 (11x faster!)
    
    DECISION RANGE:
    • 3 tiles = Can see obstacles, doors, nearby NPCs
    • 7 tiles = Can plan longer routes, see distant objects
    
    FOR DRL TRAINING:
    ✓ 7x7 is sufficient (similar to human GBA screen)
    ✓ 11x speedup enables practical training
    ✓ Agent learns navigation through exploration
    """
    
    plt.figtext(0.5, 0.02, stats_text, ha='center', fontsize=9, 
               family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.savefig('map_size_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved visualization to map_size_comparison.png")
    
    plt.show()


def print_tile_counts():
    """Print exact tile counts for different radii"""
    print("\n" + "="*60)
    print("TILE COUNT BY RADIUS")
    print("="*60)
    
    for radius in [1, 2, 3, 4, 5, 6, 7, 8, 10]:
        size = 2 * radius + 1
        tiles = size * size
        print(f"Radius {radius:2d} → {size:2d}x{size:2d} grid → {tiles:4d} tiles")
    
    print("\n" + "="*60)
    print("MEMORY READ ESTIMATES (rough)")
    print("="*60)
    print("Per tile read: ~0.2ms (metatile_id + behavior + collision)")
    print()
    
    for radius in [3, 7]:
        size = 2 * radius + 1
        tiles = size * size
        time_ms = tiles * 0.2
        print(f"{size}x{size} ({tiles:3d} tiles) → ~{time_ms:.1f}ms just for map")


if __name__ == "__main__":
    print("Creating visual comparison of map sizes...")
    visualize_map_coverage()
    print_tile_counts()
