import numpy as np

try:
    import cupy as cp
    xp = cp
except ImportError:
    import numpy as np
    xp = np

def solve_tie(I_focus, I_minus, I_plus, dz, wav_len, dx, dy):
    """
    強度輸送方程式 (TIE) を解き、位相分布 phi を算出する
    (修士論文 式 2.7 に準拠) [cite: 101]
    """
    Ny, Nx = I_focus.shape
    k = 2 * np.pi / wav_len
    
    # 1. 伝搬方向への強度微分 ∂I/∂z (式 2.6) [cite: 98]
    dI_dz = (I_plus - I_minus) / (2 * dz)
    
    # 2. フーリエ領域での周波数座標の生成 [cite: 101]
    fx = xp.fft.fftfreq(Nx, d=dx)
    fy = xp.fft.fftfreq(Ny, d=dy)
    FX, FY = xp.meshgrid(fx, fy)
    f_squared = FX**2 + FY**2
    
    # 3. 逆ラプラシアンフィルタ (式 2.7 の分母) [cite: 101]
    epsilon = 1e-10
    inv_laplacian = 1.0 / (4 * xp.pi**2 * f_squared + epsilon)
    inv_laplacian[0, 0] = 0  # 直流成分カット
    
    # --- 復元計算 ---
    # (A) ∇^-2 [ -k * ∂I/∂z ]
    term1 = xp.fft.ifft2(xp.fft.fft2(-k * dI_dz) * inv_laplacian).real
    
    # (B) 勾配 ∇ をとり I^-1 を掛ける [cite: 96]
    grad_x = xp.fft.ifft2(xp.fft.fft2(term1) * (2j * xp.pi * FX)).real
    grad_y = xp.fft.ifft2(xp.fft.fft2(term1) * (2j * xp.pi * FY)).real
    
    I_inv = 1.0 / (I_focus + epsilon)
    psi_x = I_inv * grad_x
    psi_y = I_inv * grad_y
    
    # (C) 発散 ∇・ をとり再度 逆ラプラシアンを適用 [cite: 101]
    div_psi = (xp.fft.ifft2(xp.fft.fft2(psi_x) * (2j * xp.pi * FX)) + 
               xp.fft.ifft2(xp.fft.fft2(psi_y) * (2j * xp.pi * FY))).real
    
    phi = xp.fft.ifft2(xp.fft.fft2(div_psi) * inv_laplacian).real
    
    return phi