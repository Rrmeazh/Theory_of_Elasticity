import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
import SphericalCapWithLoad

# ═════════════════════可调节参数════════════════════════
RADIUS = 0.025                            # 球冠半径 (m)
THICKNESS = 0.00015                       # 壳体厚度 (m)
CAP_ANGLE = 60                            # 球冠张角 (deg)

ELASTIC_MODULUS = 2.0e9                   # 弹性模量 (Pa)
POISSON_RATIO = 0.35                      # 泊松比

N_R = 50                                  # 径向网格点数
N_PHI = 120                               # 角向网格点数

MAX_ITER = 2000                           # 优化最大迭代次数
INDENT_RATIOS = [0.41, 0.63, 0.69, 0.83]  # 归一化压入深度 z̄ = z/h
# ══════════════════════════════════════════════════════

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
current_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(current_dir, 'output/')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def visualize(shell, w, z_ratio, n_mode):
    """
    求解结果可视化
        1. 三维表面图
        2. 位移等高线图
        3. 弯曲能量密度等高线图
        4. 环向位移剖面图

    参数:
        shell: SphericalCapWithLoad 对象
        w: 位移场数组 (m)
        z_ratio: 归一化压入深度 z̄ = z/h
        n_mode: 失稳模态数
    """
    fig = plt.figure(figsize=(16, 12))
    
    x = shell.r * np.cos(shell.PHI)
    y = shell.r * np.sin(shell.PHI)
    z = shell.z0 + w
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    norm = plt.Normalize(vmin=np.min(w), vmax=np.max(w))
    colors = cm.coolwarm(norm(w))
    ax1.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, alpha=0.9)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')    
    ax1.set_aspect('equal')

    w_apex = abs(w[:5, :].mean())
    title = f'z̄={z_ratio:.2f}, n={n_mode}, w={w_apex*1000:.3f}mm'
    ax1.set_title(title, fontsize=12, weight='bold')
    ax1.view_init(elev=25, azim=45)
    
    ax2 = fig.add_subplot(2, 2, 2)
    c2 = ax2.contourf(x, y, w * 1000, levels=50, cmap='coolwarm')
    ax2.contour(x, y, w * 1000, levels=12, colors='k', alpha=0.25, linewidths=0.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title(f'Displacement (mm), {n_mode}-fold', fontsize=11, weight='bold')
    fig.colorbar(c2, ax=ax2)
    
    ax3 = fig.add_subplot(2, 2, 3)
    U = shell.compute_energy_density(w)
    c3 = ax3.contourf(x, y, U, levels=50, cmap='hot')
    ax3.set_aspect('equal')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Bending Energy Density', fontsize=11, weight='bold')
    fig.colorbar(c3, ax=ax3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    phi_deg = np.degrees(shell.phi)
    for r_frac in [0.5, 0.7, 0.85]:
        r_idx = int(r_frac * shell.n_r)
        ax4.plot(phi_deg, w[r_idx, :] * 1000, 
                label=f'r/a = {shell.rho[r_idx]:.2f}', lw=2)
    ax4.set_xlabel('φ (degrees)')
    ax4.set_ylabel('w (mm)')
    ax4.set_title('Circumferential Profiles', fontsize=11, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', ls='--', alpha=0.5)
    ax4.set_xlim([0, 360])
    
    plt.tight_layout()
    fname = f'buckling_z{z_ratio:.2f}_n{n_mode}.png'
    plt.savefig(OUTPUT_DIR + fname, dpi=200)
    plt.show()

def main():
    print("="*70)
    print(" Spherical Cap Buckling - Displacement Control")
    print(" Key: Using displacement as control parameter")
    print("="*70)
    
    shell = SphericalCapWithLoad.SphericalCapWithLoad(
        radius=RADIUS,
        thickness=THICKNESS,
        cap_angle=CAP_ANGLE,
        elastic_modulus=ELASTIC_MODULUS,
        poisson_ratio=POISSON_RATIO,
        n_r=N_R,
        n_phi=N_PHI,
        max_iter=MAX_ITER
    )
    
    for z_ratio in INDENT_RATIOS:
        w, n_mode = shell.solve(z_ratio)
        w_apex = w[:5, :].mean()
        print(f"\n  z̄ = {z_ratio:.2f}")
        print(f"  w_apex = {abs(w_apex)*1000:.3f} mm")
        print(f"  Mode: n = {n_mode}")
        visualize(shell, w, z_ratio, n_mode)
    
    print("\nDone!")
    return shell, w


if __name__ == "__main__":
    shell, w = main()