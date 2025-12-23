try:
    import cupy as cp
    xp = cp
    use_gpu = True
except ImportError:
    import numpy as np
    xp = np
    use_gpu = False

class Propagation:
    def __init__(self, wav_len, dx, dy):
        self.wav_len = wav_len
        self.dx = dx
        self.dy = dy
        self.use_gpu = use_gpu

    def nearprop_conv(self, Comp1, sizex, sizey, d, return_gpu=False):
        # Transfer to GPU if available
        if use_gpu:
            comp1_gpu = cp.asarray(Comp1)
        else:
            comp1_gpu = xp.array(Comp1)
        
        if d == 0:
            if return_gpu and use_gpu:
                return comp1_gpu
            return Comp1
            
        x1, x2 = -sizex // 2, sizex // 2 - 1
        y1, y2 = -sizey // 2, sizey // 2 - 1
        
        # Grid
        Fx, Fy = xp.meshgrid(xp.arange(x1, x2+1), xp.arange(y1, y2+1))
        
        # FFT
        Fcomp1 = xp.fft.fftshift(xp.fft.fft2(comp1_gpu)) / xp.sqrt(sizex * sizey)
        
        # Phase factor
        FresR = xp.exp(-1j * xp.pi * self.wav_len * d * ((Fx**2) / ((self.dx * sizex)**2) + (Fy**2) / ((self.dy * sizey)**2)))
        Fcomp2 = Fcomp1 * FresR
        
        # IFFT
        res_gpu = xp.fft.ifft2(xp.fft.ifftshift(Fcomp2)) * xp.sqrt(sizex * sizey)
        
        # Transfer back to CPU only if requested
        if use_gpu and not return_gpu:
            return cp.asnumpy(res_gpu)
        else:
            return res_gpu
