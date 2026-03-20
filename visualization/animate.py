import struct
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys
import os
import math
import random

# --- Configuration ---
BIN_FILE = "galaxy_data.bin"       
GIF_FILE = "docs/assets/simulation_demo.gif"
G = 1.0            
DT = 0.01          
SOFTENING = 0.1    
FRAMES = 1000       
GALAXY_INDEX = 0   
VIEW_RADIUS = 250

# --- Visual & Performance Configuration ---
FIG_WIDTH = 9.5          # 9.5 inches * 100 DPI = 950 pixels wide
FIG_HEIGHT = 5.0         # 5.0 inches * 100 DPI = 500 pixels tall
DPI = 100                # Resolution multiplier
CPU_MAX_BODIES = 4000    # Increased from 1500. Less downsampling = denser galaxy!

# --- Ring Size Configuration ---
RING_INNER_DIST = 43.0   
RING_OUTER_DIST = 300.0  
NUM_SYSTEMS = 32
BODIES_PER_SYSTEM = 8192

# Use GPU if available, otherwise fallback to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using compute device: {device}")

# ==========================================
# 1. DATA GENERATION
# ==========================================
def generate_stable_system(offset_base):
    bodies = []
    center_mass = 1000.0
    
    bodies.append({
        'x': 0.0, 'y': 0.0, 'z': 0.0,
        'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
        'mass': center_mass
    })
    
    for _ in range(BODIES_PER_SYSTEM - 1):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(RING_INNER_DIST, RING_OUTER_DIST)  
        velocity = math.sqrt(center_mass / dist)
        
        x = dist * math.cos(angle)
        y = dist * math.sin(angle)
        z = random.uniform(-10, 10)
        
        vx = -velocity * math.sin(angle)
        vy = velocity * math.cos(angle)
        vz = 0.0
        
        bodies.append({
            'x': x, 'y': y, 'z': z,
            'vx': vx, 'vy': vy, 'vz': vz,
            'mass': 1.0
        })
    return bodies

def generate_binary_file(filename):
    print(f"Generating galaxy data into {filename}...")
    with open(filename, 'wb') as f:
        f.write(struct.pack('ii', NUM_SYSTEMS, BODIES_PER_SYSTEM))
        for i in range(NUM_SYSTEMS):
            system_data = generate_stable_system(i)
            for b in system_data:
                data = struct.pack('ffffff', b['x'], b['y'], b['z'], b['vx'], b['vy'], b['vz'])
                f.write(data)
    print("✅ Generation complete.")

# ==========================================
# 2. PHYSICS & ANIMATION
# ==========================================
def read_bin_snapshot(filename, target_galaxy):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        sys.exit(1)

    with open(filename, 'rb') as f:
        header = f.read(8)
        num_systems, bodies_per_system = struct.unpack('ii', header)
        
        bytes_per_body = 6 * 4 
        offset = 8 + (target_galaxy * bodies_per_system * bytes_per_body)
        
        f.seek(offset)
        raw_data = f.read(bodies_per_system * bytes_per_body)
        data = np.frombuffer(raw_data, dtype=np.float32).reshape((bodies_per_system, 6)).copy()
        
        # --- CPU DOWNSAMPLING (UPDATED) ---
        if device.type == 'cpu' and bodies_per_system > CPU_MAX_BODIES:
            print(f"⚠️ CPU detected. Downsampling to {CPU_MAX_BODIES} bodies for rendering speed...")
            indices = np.random.choice(range(1, bodies_per_system), CPU_MAX_BODIES - 1, replace=False)
            indices = np.insert(indices, 0, 0) 
            data = data[indices]
            bodies_per_system = CPU_MAX_BODIES
            
        pos = data[:, 0:3]
        vel = data[:, 3:6]
        masses = np.ones(bodies_per_system, dtype=np.float32)
        masses[0] = 1000.0 
        
        pos_t = torch.tensor(pos, device=device)
        vel_t = torch.tensor(vel, device=device)
        masses_t = torch.tensor(masses, device=device)
        
        return pos_t, vel_t, masses_t

def compute_gravity_step(pos, vel, masses):
    dx = pos.unsqueeze(1) - pos.unsqueeze(0)
    dist_sq = torch.sum(dx**2, dim=-1) + SOFTENING**2
    inv_dist_cube = dist_sq**(-1.5)
    inv_dist_cube.fill_diagonal_(0.0) 
    acc = torch.sum(dx * (masses.unsqueeze(1) * inv_dist_cube).unsqueeze(-1), dim=0) * G
    vel -= acc * DT
    pos += vel * DT
    return pos, vel

def main():
    if not os.path.exists(BIN_FILE):
        generate_binary_file(BIN_FILE)
        
    print(f"Loading data from {BIN_FILE}...")
    pos, vel, masses = read_bin_snapshot(BIN_FILE, GALAXY_INDEX)
    
    # --- Set up Widescreen 3D Matplotlib Figure ---
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    fig.patch.set_facecolor('black')
    
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.axis('off')
    ax.grid(False)
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

    pos_cpu = pos.cpu().numpy()
    
    num_stars = len(pos_cpu) - 1
    star_sizes = np.random.uniform(0.1, 3.0, size=num_stars)
    
    scatter = ax.scatter(pos_cpu[1:, 0], pos_cpu[1:, 1], pos_cpu[1:, 2], 
                         s=star_sizes, color='white', alpha=0.9, edgecolors='none')

    scatter_glow = ax.scatter(pos_cpu[0:1, 0], pos_cpu[0:1, 1], pos_cpu[0:1, 2], 
                              s=400, color='white', alpha=0.08, edgecolors='none')

    ax.text2D(0.02, 0.96, 'N-Body Simulation\nCUDA Multi-GPU Acceleration', 
              transform=ax.transAxes, color='white', alpha=0.7, 
              fontsize=11, verticalalignment='top', fontfamily='monospace')

    iteration_text = ax.text2D(0.5, 0.04, 'Iteration: 0', 
                               transform=ax.transAxes, color='white', alpha=0.9, 
                               fontsize=11, horizontalalignment='center', fontfamily='monospace')

    def update(frame):
        nonlocal pos, vel
        pos, vel = compute_gravity_step(pos, vel, masses)
        
        pos_cpu = pos.cpu().numpy()
        
        scatter._offsets3d = (pos_cpu[1:, 0], pos_cpu[1:, 1], pos_cpu[1:, 2])
        scatter_glow._offsets3d = (pos_cpu[0:1, 0], pos_cpu[0:1, 1], pos_cpu[0:1, 2])
        
        bh_pos = pos_cpu[0]
        ax.set_xlim(bh_pos[0] - VIEW_RADIUS, bh_pos[0] + VIEW_RADIUS)
        ax.set_ylim(bh_pos[1] - VIEW_RADIUS, bh_pos[1] + VIEW_RADIUS)
        ax.set_zlim(bh_pos[2] - VIEW_RADIUS, bh_pos[2] + VIEW_RADIUS)
        
        # Cinematic Pan
        ax.view_init(elev=25, azim=45 + (frame * 0.25))

        iteration_text.set_text(f'Iteration: {frame}')
        
        if frame % 10 == 0:
            print(f"Rendering frame {frame}/{FRAMES}...")
            
        return scatter, scatter_glow, iteration_text

    anim = animation.FuncAnimation(fig, update, frames=FRAMES, interval=30, blit=False)

    os.makedirs(os.path.dirname(GIF_FILE), exist_ok=True)
    print(f"Generating 950x500 Widescreen GIF...")
    
    anim.save(GIF_FILE, writer='pillow', fps=24)
    print(f"✅ GIF successfully saved to {GIF_FILE}")

    plt.show()

if __name__ == "__main__":
    main()