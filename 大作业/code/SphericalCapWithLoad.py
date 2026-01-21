import numpy as np
from scipy.optimize import minimize

class SphericalCapWithLoad:
    def __init__(self, radius, thickness, cap_angle, elastic_modulus, poisson_ratio, n_r=60, n_phi=120, max_iter=3000):
        """
        球冠几何参数、材料参数与网格初始化

        参数:
            radius: 球冠半径 (m)
            thickness: 壳体厚度 (m)
            cap_angle: 球冠张角 (deg)
            elastic_modulus: 弹性模量 (Pa)
            poisson_ratio: 泊松比
            n_r: 径向网格点数
            n_phi: 周向网格点数
            max_iter: 优化最大迭代次数
        """
        self.R = radius
        self.t = thickness
        self.alpha = np.radians(cap_angle)
        self.a = self.R * np.sin(self.alpha)
        self.h = self.R * (1 - np.cos(self.alpha))
        self.l_c = np.sqrt(self.R * self.t) # 弯曲-膜过渡长度尺度

        self.E = elastic_modulus
        self.nu = poisson_ratio
        self.D = self.E * self.t**3 / (12 * (1 - self.nu**2)) # 弯曲刚度
        self.K = self.E * self.t / (1 - self.nu**2) # 膜刚度
        
        self.n_r = n_r
        self.n_phi = n_phi
        self.max_iter = max_iter
        
        self.rho = np.linspace(0.01, 1.0, self.n_r)
        self.phi = np.linspace(0, 2*np.pi, self.n_phi, endpoint=False)
        self.RHO, self.PHI = np.meshgrid(self.rho, self.phi, indexing='ij')
        
        # 网格微元
        self.dr = self.rho[1] - self.rho[0]
        self.dphi = self.phi[1] - self.phi[0]
        
        # 计算网格坐标和面积微元
        self.r = self.RHO * self.a
        self.z0 = self.h - (self.r**2) / (2 * self.R)
        self.dA = self.r * self.dr * self.a * self.dphi
    
        
    def compute_total_potential(self, w_flat, z_target):
        """
        计算总势能，包括：
            球壳的弯曲能和膜应变能
            外力功：位移控制中，以弹簧势能形式表示
            位移约束能：用弹簧势能形式加入位移目标
            边界约束能：以正则项形式加入边界约束
        
        参数:
            w_flat: 挠度场，一维数组
            z_target: 目标顶点位移
        返回:
            total_potential: 总势能值
        """
        w = w_flat.reshape(self.n_r, self.n_phi)
        dr = self.dr * self.a
        dphi = self.dphi
        r = self.r + 1e-10
        
        # ==== 挠度关于坐标的偏导数 =====
        w_r = np.zeros_like(w)
        w_r[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2 * dr)
        w_r[0, :] = (w[1, :] - w[0, :]) / dr
        w_r[-1, :] = 0
        
        w_rr = np.zeros_like(w)
        w_rr[1:-1, :] = (w[2:, :] - 2*w[1:-1, :] + w[:-2, :]) / dr**2
        w_rr[0, :] = 2 * (w[1, :] - w[0, :]) / dr**2
        w_rr[-1, :] = 0
        
        w_phi = np.zeros_like(w)
        w_phi[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * dphi)
        w_phi[:, 0] = (w[:, 1] - w[:, -1]) / (2 * dphi)
        w_phi[:, -1] = (w[:, 0] - w[:, -2]) / (2 * dphi)
        
        w_phiphi = np.zeros_like(w)
        w_phiphi[:, 1:-1] = (w[:, 2:] - 2*w[:, 1:-1] + w[:, :-2]) / dphi**2
        w_phiphi[:, 0] = (w[:, 1] - 2*w[:, 0] + w[:, -1]) / dphi**2
        w_phiphi[:, -1] = (w[:, 0] - 2*w[:, -1] + w[:, -2]) / dphi**2
        
        w_rphi = np.zeros_like(w)
        w_rphi[:, 1:-1] = (w_r[:, 2:] - w_r[:, :-2]) / (2 * dphi)
        w_rphi[:, 0] = (w_r[:, 1] - w_r[:, -1]) / (2 * dphi)
        w_rphi[:, -1] = (w_r[:, 0] - w_r[:, -2]) / (2 * dphi)
        # =============================
        
        # ========== 球壳曲率 ==========
        kappa_r = w_rr
        kappa_phi = w_r / r + w_phiphi / r**2
        kappa_rphi = w_rphi / r - w_phi / r**2
        # =============================
        
        # =========== 膜应变 ===========
        eps_r = w / self.R + 0.5 * w_r**2
        eps_phi = w / self.R + 0.5 * (w_phi / r)**2
        gamma_rphi = w_r * w_phi / r
        # =============================
        
        # 1.计算弯曲能、膜应变能
        U_bend = 0.5 * self.D * (
            kappa_r**2 + kappa_phi**2 +
            2 * self.nu * kappa_r * kappa_phi +
            2 * (1 - self.nu) * kappa_rphi**2
        )
        
        membrane_factor = 0.3
        U_memb = membrane_factor * 0.5 * self.K * (
            eps_r**2 + eps_phi**2 +
            2 * self.nu * eps_r * eps_phi +
            0.5 * (1 - self.nu) * gamma_rphi**2
        )
        
        E_strain = np.sum((U_bend + U_memb) * self.dA)

        # 2.计算外力功、位移约束能
        w_apex = w[:5, :].mean()
        k_spring = 5 * self.K * self.a
        E_force = - k_spring * w_apex **2 / 2
        E_constraint = k_spring * (w_apex - z_target)**2
        
        # 3.计算边界约束能
        w_edge = w[-1, :]
        w_edge_slope = (w[-1, :] - w[-2, :]) / dr
        
        k_edge = 5 * self.D / self.a**2
        E_edge = 0.5 * k_edge * (
            np.sum(w_edge**2) * self.a * self.dphi +
            np.sum(w_edge_slope**2) * self.a * self.dphi
        )
        
        total_potential = E_strain + E_constraint - E_force + E_edge
        return total_potential
    
    
    def create_edge_localized_initial(self, z_target, n_mode):
        """
        创建边缘局域化初始猜测，在轴对称基础上叠加边缘局域化扰动

        参数:
            z_target: 目标顶点位移
            n_mode: 边缘扰动模态数
        返回:
            w0: 初始挠度场
        """
        w_est = abs(z_target)
        w_base = z_target * np.exp(-1.0 * (self.RHO / 0.85)**2)
        
        # 扰动幅度和相位
        depth_ratio = w_est / self.h
        base_amp = 0.25 * w_est
        mode_bias = (n_mode - 4.5) * 0.05 * depth_ratio
        amp = base_amp * (1 + mode_bias)
        
        phase = np.random.uniform(0, 2*np.pi / n_mode)
        
        # 边缘包络
        r_center = 0.78
        r_width = 0.18
        r_envelope = np.exp(-((self.RHO - r_center) / r_width)**2)
        
        perturbation = amp * r_envelope * np.cos(n_mode * self.PHI + phase)
        
        w0 = w_base + perturbation
        w0[-1, :] = 0
        w0[-2, :] = 0
        return w0
    
    
    def solve(self, z_ratio):
        """
        通过多模态初始猜测和能量最小化，找到最低能量解。通过检测模态数和边界一致性筛选有效解。最终返回最低能量解的挠度场和模态数。

        参数:
            z_ratio: 目标顶点位移与高度之比 (z/h)
        返回:
            best_w: 最优挠度场
            best_n: 最优模态数
        """
        z_target = -z_ratio * self.h  # 向下为负
        print(f"\nSolving with z̄ = {z_ratio:.2f} (z = {abs(z_target)*1000:.3f} mm)")
        
        test_modes = list(range(3, 9))
        print(f"Testing all modes: {test_modes}")
        
        best_energy = np.inf
        best_w = None
        best_n = 0
        
        for n_mode in test_modes:
            for attempt in range(3):
                np.random.seed(n_mode * 100 + attempt)
                
                w0 = self.create_edge_localized_initial(z_target, n_mode)
                
                result = minimize(
                    self.compute_total_potential,
                    w0.flatten(),
                    args=(z_target,),
                    method='L-BFGS-B',
                    options={
                        'maxiter': self.max_iter,
                        'ftol': 1e-10,
                        'gtol': 1e-7
                    }
                )
                
                w_temp = result.x.reshape(self.n_r, self.n_phi)
                n_detected = self.detect_mode_number(w_temp)
                edge_diff = np.max(np.abs(w_temp[:, 0] - w_temp[:, -1]))
                
                if attempt == 0:
                    print(f"  Testing n={n_mode}...")
                
                if result.fun < best_energy and edge_diff < 1e-5 and abs(n_detected - n_mode) <= 1:
                    best_energy = result.fun
                    best_w = w_temp
                    best_n = n_detected
        
        if best_w is None:
            best_w = w_temp
            best_n = self.detect_mode_number(best_w)
        
        w_apex = best_w[:5, :].mean()
        print(f"  FINAL: n={best_n}, Π={best_energy:.5e}, w={abs(w_apex)*1000:.3f}mm")
        return best_w, best_n
    
    
    def detect_mode_number(self, w):
        """
        通过FFT分析检测边缘模态数。取不同径向位置的截面，计算其FFT谱，寻找主导模态数。采用中值作为最终模态数输出。

        参数:
            w: 挠度场
        返回:
            n_mode: 检测到的模态数
        """
        modes = []
        
        for r_frac in [0.70, 0.75, 0.80, 0.85]:
            r_idx = int(r_frac * self.n_r)
            if r_idx >= self.n_r:
                continue
            
            w_circ = w[r_idx, :] - w[r_idx, :].mean()
            fft = np.abs(np.fft.fft(w_circ))
            power = fft**2
            
            n_max = min(20, self.n_phi // 2)
            peak_idx = np.argmax(power[1:n_max]) + 1
            
            if power[peak_idx] > 0.15 * power[0]:
                modes.append(peak_idx)
        
        if len(modes) >= 2:
            return int(np.round(np.median(modes)))
        elif len(modes) == 1:
            return modes[0]
        else:
            return 0
    
    
    def compute_energy_density(self, w):
        """
        计算弯曲能密度分布

        参数:
            w: 挠度场
        返回:
            U: 归一化弯曲能密度分布
        """
        dr = self.dr * self.a
        r = self.r + 1e-10
        
        w_r = np.gradient(w, dr, axis=0)
        w_rr = np.gradient(w_r, dr, axis=0)
        
        w_phi = np.zeros_like(w)
        w_phi[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2 * self.dphi)
        w_phi[:, 0] = (w[:, 1] - w[:, -1]) / (2 * self.dphi)
        w_phi[:, -1] = (w[:, 0] - w[:, -2]) / (2 * self.dphi)
        
        w_phiphi = np.zeros_like(w)
        w_phiphi[:, 1:-1] = (w[:, 2:] - 2*w[:, 1:-1] + w[:, :-2]) / self.dphi**2
        w_phiphi[:, 0] = (w[:, 1] - 2*w[:, 0] + w[:, -1]) / self.dphi**2
        w_phiphi[:, -1] = (w[:, 0] - 2*w[:, -1] + w[:, -2]) / self.dphi**2
        
        kappa_r = -w_rr
        kappa_phi = -(w_r / r + w_phiphi / r**2)
        
        U = self.D * (kappa_r**2 + kappa_phi**2 + 2 * self.nu * kappa_r * kappa_phi)
        return U / (np.max(U) + 1e-20)