import numpy as np
import scipy as sp
from scipy.integrate._ivp.base import OdeSolver
from scipy.integrate._ivp.rk import rk_step
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as plticker
import matplotlib.gridspec as gs 
import pickle
mpl.rcParams.update({'font.family': 'CMU Sans Serif', 'axes.unicode_minus' : False})
mpl.rcParams.update({'mathtext.fontset': 'cm', 'mathtext.default':'it'})
import sys
from dataclasses import dataclass
np.seterr(over = 'raise', under = 'ignore')
from wa_phase_diagram import phase_portrait, temporal_trace, stability_diagram
from reduced_model_utils import *

@dataclass
class Sol:
    t: np.ndarray
    y: np.ndarray

def rk_step(fun, t, y, f, h, A, B, C, cap = 0):
    K = np.empty((6, len(y)))
    K[0] = f
    for s, (a, c) in enumerate(zip(A[1:], C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy)

    y_new = y + h * np.dot(K.T, B)

    if y_new[0] < 0:
        y_new[0] = 0

    w = y_new[2:].copy()
    if cap > 0:
        w[w>cap] = cap
    w[w<0] = 0
    y_new[2:] = w

    f_new = fun(t+h, y_new)

    return y_new, f_new

def solve_ivp(fun, t, y0, cap = 0):
    A = np.array([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])

    def fun_single(t, y):
        return fun(t, y[:,None]).ravel()

    ny = len(y0)
    sol = Sol(t, np.empty((ny,len(t)), dtype = float))
    sol.y[:] = np.nan
    y_new = y0
    f_new = fun_single(t[0], y_new)
    sol.y[:,0] = y0
    for i in range(1,len(t)):
        h = t[i] - t[i-1]
        y_new, f_new = rk_step(fun_single, t[i-1], y_new, f_new, h, A, B, C, cap)
        sol.y[:,i] = y_new

    return sol


old_init = OdeSolver.__init__
old_step = OdeSolver.step

def new_step(self):
    message = old_step(self)
    return message

def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):
    old_init(self, fun, t0, y0, t_bound, vectorized, support_complex)

# RF pool shape
def half_square(x, l):
    return l/2
def iint_square(x, l):
    return x*l
def int_square(t,v=3.2,d=3,l=8,nd=2):
    bot, top = get_xrange(t,v,d,l,nd)
    return iint_square(top, l) - iint_square(bot, l)

def half_circle_height(x, l):
    return np.sqrt(l*l - np.power(x - l,2))

def iint_circle_chord(x, l):
    return 2*l*l*np.arcsin(np.sqrt(x/(2*l))) + (x - l)*half_circle_height(x,l)

def int_circle_chord(t, v= 3.2, d = 3, l = 8, nd = 2):
    bot, top = get_xrange(t,v,d,l,nd)
    return iint_circle_chord(top, l) - iint_circle_chord(bot, l)

def int_func(pt, w, fxt, x, l):
    return 2*np.interp(pt, x, w)*fxt(pt,l)

def int_fxt_and_W(w, uniform_w, iint_fxt, t, l = 8, d = 3, v= 3.2, nd = 2, quad = False):
    bot, top = get_xrange(t,v,d,l,nd)
        
    if uniform_w:
        if hasattr(w, '__len__'):
            integral = (iint_fxt(top, l) - iint_fxt(bot, l))*w[0]
            if len(w.shape) == 1:
                return integral
            else:
                return integral*np.ones_like(w)
        else:
            return (iint_fxt(top, l) - iint_fxt(bot, l))*w
    else:
        if len(w.shape) == 2:
            nx = w.shape[0]
        else:
            nx = len(w)
            w = np.reshape(w, (nx, 1))

        discrete = np.zeros(w.shape[1])
        x = np.linspace(0,2*l,nx)
        if quad:
            fxt = iint_fxt
            for i in range(w.shape[1]):
                discrete[i] = sp.integrate.quad(int_func, bot, top, args = (w[:,i], fxt, x, l))[0]

        else:
            for i in range(w.shape[1]):
                for j in range(0,nx-1): # forward scheme
                    # whole dx
                    if bot <= x[j] and x[j+1] <= top:
                        discrete[i] += (w[j,i] + w[j+1,i])/2*(iint_fxt(x[j+1], l) - iint_fxt(x[j], l))
                    
                    #  head 
                    elif x[j] < bot and bot < x[j+1]:
                        w_prime = w[j,i] + (w[j+1,i]-w[j,i]) * (bot - x[j])/(x[j+1] - x[j])
                        discrete[i] += (w_prime + w[j+1,i])/2 * (iint_fxt(x[j+1], l) - iint_fxt(bot, l))
                    # tail
                    elif x[j] < top and top < x[j+1]:
                        w_prime = w[j,i] + (w[j+1,i]-w[j,i]) * (top-x[j])/(x[j+1] - x[j])
                        discrete[i] += (w[j,i] + w_prime)/2 * (iint_fxt(top, l) - iint_fxt(x[j], l))

        #if discrete!=0:
        #    print(t, discrete, top-bot, l, d, v, nx, w.T)
        if len(w.shape) == 2:
            return discrete
        else:
            return discrete[0]

def dInput_approx(r, r_bar, w, uniform_w, half_fxt, iint_fxt, t, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, F_perc = 1.0):
    # approx: w fixed during wave
    Gt, bot, top = deltaW_Area(r, r_bar, iint_fxt, t, k, l, d, v, r0, rx)
    Ft = F_perc * deltaArea_W(half_fxt, w, uniform_w, rx, v, l, bot = bot, top = top)
    return Gt + Ft

def dR(r, w, uniform_w, iint_fxt, t, dt = None, l = 8, d = 3, v = 3.2, rx = 20, gL = 40, tau_m = 0.02, quad = False):
    #print(r, gL, tau_m)
    #print(np.mean(w), uniform_w)
    dr = (-gL*r + rx*int_fxt_and_W(w,uniform_w,iint_fxt,t,l=l,d=d,v=v,quad=quad))/tau_m
    #print(f'dr = {dr}')
    #if dt is not None:
    #    pick = r + dr*dt < 0
    #    dr[pick] = -r[pick]/dt
    #print(f'dr = {dr}')
    return dr

def dR_approx2(r, r_bar, w, uniform_w, half_fxt, iint_fxt, t, dt = None, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gL = 40, F_perc = 1.0):
    dI = dInput_approx(r, r_bar, w, uniform_w, half_fxt, iint_fxt, t, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, F_perc = F_perc)
    if dt is not None:
        pick = r + dI*dt < 0
        dI[pick] = -r[pick]/dt
    return dI/gL

def dR_approx3(dr, dt = None, r = None):
    if dt is None:
        return dr
    else:
        _dr = dr.copy()
        pick = r + dr*dt < 0
        _dr[pick] = -r[pick]/dt
        return _dr

def dRbar(r_bar, r, tau = 1.0):
    return (r - r_bar)/tau

def _dW(r, rbar, k, rx, r0):
    return  k*rx*r*(r-rbar/r0*rbar)

def dW(r, rbar, k,l,d,v,r0,rx,t,nx,dt,w, nd = 2, cap = 0, skip_LGN = None):
    if w is not None:
        dw = np.zeros_like(w)
    else:
        dw = np.zeros((nx,1))

    bot, top = get_xrange(t,v,d,l,nd)
    x = np.linspace(0,2*l,nx)
    pick = np.logical_and(x <= top, x >= bot)
    dw[pick,:] = _dW(r,rbar, k,rx,r0)
    pick = w + dw*dt < 0 
    dw[pick] = -w[pick]/dt
    if cap > 0:
        pick = w + dw*dt > cap 
        dw[pick] = (cap - w[pick])/dt
    if skip_LGN is not None:
        dw[skip_LGN] = 0
    return dw

def nullcline1(w, uniform_w, iint_fxt, t, l, d, v, rx, gL, quad = False):
    return rx*int_fxt_and_W(w,uniform_w,iint_fxt,t,l,d,v,quad)/gL

def get_dF1(w, uniform_w, iint_fxt, t, l = 8, d = 3, v = 3.2, rx = 20, gL = 40, tau_m = 0.02, tau = 1, quad = False):
    def dF(r, r_bar):
        return dR(r, w, uniform_w, iint_fxt, t, l=l,d=d,v=v,rx=rx,gL=gL,tau_m=tau_m,quad=quad), dRbar(r_bar, r, tau = tau)
    return dF

def next_W(t, w0, r, rbar, k,l,d,v,rx,r0, nd = 2, skip_LGN = None, end_only = True): # euler
    nx = w0.size
    nt = t.size
    x = np.linspace(0,2*l,nx)
    Dt = (np.sqrt(2)-1)*l/v
    left = x
    right = x+d*nd
    if nd == 1:
        left += d*nd
        right += d*nd
    left = left/v + Dt
    right = right/v + Dt
    w = np.zeros((nx,nt))
    w[:,0] = w0
    delta = np.zeros((nx,nt-1))
    for i in range(1,nt):
        dw0 = _dW(r[i-1], rbar[i-1], k,rx,r0)
        dw1 = _dW(r[i],rbar[i], k,rx,r0) 
        dt = t[i] - t[i-1]
        pick0 = np.logical_and(left < t[i], t[i] < right)
        
        pick = np.logical_and(pick0, t[i-1] < left)
        if sum(pick) > 0:
            dw_prime = dw0 + (dw1-dw0) * (left[pick] - t[i-1])/dt
            delta[pick,i-1] = (dw_prime + dw1)/2 * (t[i] - left[pick])
            
        pick = np.logical_and(pick0, left <= t[i-1])
        if sum(pick) > 0:
            delta[pick,i-1] = (dw0 + dw1)/2 * (t[i] - t[i-1])
            
        pick = np.logical_and(t[i-1] < right, right < t[i])
        if sum(pick) > 0:
            ddt = right[pick]-t[i-1]
            dw_prime = dw0 + (dw1-dw0) * ddt/dt
            delta[pick,i-1] = (dw0 + dw_prime)/2 * ddt
            
    w[:,1:] = (w0 + np.cumsum(delta, axis = -1).T).T
    w[w<0] = 0
    if skip_LGN is not None:
        w[skip_LGN] = 0
    if end_only:
        return w[:,-1]
    else:
        return w
    
def get_exact0(iint_fxt, dt, cap = 0, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gL = 40, tau_m = 0.02, tau = 1, quad = False):
    def func(t, y):
        y_next = np.zeros_like(y)
        y_next[0,:] = dR(y[0,:], y[2:,:], False, iint_fxt, t, dt=dt, l=l, d=d, v=v, rx=rx, gL=gL, tau_m=tau_m,quad=quad)
        y_next[1,:] = dRbar(y[1,:], y[0,:], tau = tau)
        y_next[2:,:] = dW(y[0,:], y[1,:], k,l,d,v,r0,rx,t,y.shape[0]-2,dt,y[2:,:], cap = cap)
        return y_next
    return func

def R(w, uniform_w, iint_fxt, t, gain, l = 8, d = 3, v = 3.2, rx = 20, quad = False):
    return gain*rx*int_fxt_and_W(w, uniform_w, iint_fxt, t, l=l, d=d, v=v, quad=quad)

def get_exact1(iint_fxt, dt, cap = 0, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gain = 0.1, tau = 1, quad = False):
    def func(t, y):
        y_next = np.zeros_like(y)
        r = R(y[2:,:], False, iint_fxt, t, gain, l=l, d=d, v=v, rx=rx, quad = quad)
        y_next[0,:] = (r - y[0,:])/dt
        y_next[1,:] = dRbar(y[1,:], r, tau = tau)
        y_next[2:,:] = dW(r, y[1,:], k,l,d,v,r0,rx,t,y.shape[0]-2,dt,y[2:,:], cap = cap)
        return y_next
    return func

def get_exact2(half_fxt, iint_fxt, dt, cap = 0, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gain = 0.1, tau = 1, quad = False, F_perc = 1.0):
    def func(t, y):
        y_next = np.zeros_like(y)
        y_next[0,:] = dR_approx2(y[0,:], y[1,:], y[2:,:], False, half_fxt, iint_fxt, t, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = 1/gain, F_perc = F_perc) # R'
        y_next[1,:] = dRbar(y[1,:], y[0,:], tau = tau)
        y_next[2:,:] = dW(y[0,:], y[1,:], k,l,d,v,r0,rx,t,y.shape[0]-2,dt,y[2:,:], cap = cap)
        return y_next
    return func
   
def get_approx1(w, uniform_w, iint_fxt, dt = None, l = 8, d = 3, v = 3.2, rx = 20, gL = 40, tau_m = 0.02, tau = 1, quad = False):
    def func(t, y):
        y_next = np.zeros_like(y)
        y_next[0,:] = dR(y[0,:], w, uniform_w, iint_fxt, t, dt=dt, l=l,d=d,v=v,rx=rx,gL=gL,tau_m=tau_m,quad=quad)
        y_next[1,:] = dRbar(y[1,:], y[0,:], tau = tau)
        return y_next
    return func
    
def get_approx2(w, uniform_w, half_fxt, iint_fxt, dt = None, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 1, gL = 40, tau = 1, F_perc = 1.0):
    def func(t, y):
        y_next = np.zeros_like(y)
        y_next[0,:] = dR_approx2(y[0,:], y[1,:], w, uniform_w, half_fxt, iint_fxt, t, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, F_perc = F_perc) # R'
        y_next[1,:] = dRbar(y[1,:], y[0,:], tau = tau) # R_bar'
        return y_next
    return func

def get_approx3(w, uniform_w, half_fxt, iint_fxt, dt = None, k = 0.114*0.017, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gL = 40, tau_m = 0.02, tau = 1, print_out = False, F_perc = 1.0):
    def func(t, y):
        y_next = np.zeros_like(y)
        y_next[0,:] = dR_approx3(y[2,:], dt = dt, r = y[0,:]) # R'
        y_next[1,:] = dRbar(y[1,:], y[0,:], tau = tau) # R_bar'
        y_next[2,:] = (dInput_approx(y[0,:], y[1,:], w, uniform_w, half_fxt, iint_fxt, t, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, F_perc = F_perc) - gL*y[2,:])/tau_m # R''
        if print_out:
            print(f't={t}: y={y.T}, dy={y_next.T}')
        return y_next
    return func

def show_w_over_t(wt, dt, tlabel, wlabel, l, T = 0, ax = None, cmap = 'Greys_r', log = False, fs = 'large'):
    nt = wt.shape[1]
    nx = wt.shape[0]
    x = np.linspace(0, 2*l, nx)
    pick = np.var(wt, axis = 0) > np.finfo(float).eps
    imax = np.argmax(wt, axis = 0)
    if ax is None:
        ax = [None]*2
        if T == 0:
            fig = plt.figure(figsize = (8,3))
            ax[0] = fig.add_subplot(1,3,(1,2))
            ax[1] = fig.add_subplot(1,3,3)
        else:
            nT = int(np.round(nt/T))
            fig = plt.figure(figsize = (nT*4,3))
            ax[0] = fig.add_subplot(1,nT+1,(1,nT))
            ax[1] = fig.add_subplot(1,nT+1,nT+1)
        ax_is_None = True
    else:
        ax_is_None = False
        
    _wt = wt.copy()
    #_wt[wt == 0] = np.nan
    if log:
        im = ax[0].imshow(np.log(_wt), origin = 'lower', cmap = cmap)
        #plt.colorbar(im, shrink = 0.9, label = f'ln({wlabel})')
    else:
        im = ax[0].imshow(_wt, origin = 'lower', cmap = cmap)
        #plt.colorbar(im, shrink = 0.9, label = wlabel)
    ax[0].plot(np.arange(nt)[pick], imax[pick], 'r')
    if T > 0:
        for i in np.arange(0, nt, T):
            ax[0].plot([i, i], [0,nx], ':b')
    xtick = ax[0].get_xticks()
    ax[0].set_xticks(xtick, labels = [f'{tick*dt:.0f}' if int(tick) == tick else f'{tick*dt:.1f}' for tick in xtick ])
    ytick = ax[0].get_yticks()
    ax[0].set_yticks(ytick, labels = [f'{tick/(nx-1)*(2*l):.0f}' if int(tick/(nx-1)*(2*l)) == tick/(nx-1)*(2*l) else f'{tick/(nx-1)*(2*l):.1f}' for tick in ytick])
    ax[0].set_xlim(-0.5, nt-0.5)
    ax[0].set_ylim(-0.5, nx-0.5)
    ax[0].set_xlabel(tlabel, fontsize = fs)
    ax[0].set_ylabel(r'$x$', fontsize = fs)
    ax[0].set_aspect('auto')
    cb = plt.colorbar(im, orientation = 'horizontal', location = 'top', shrink = 0.9)

    chartBox = ax[0].get_position()
    ax[0].set_position([chartBox.x0, chartBox.y0, chartBox.width*1.44, chartBox.height*0.85])

    cb.ax.text(1.5, 1.7, wlabel, fontsize = 'medium', transform = cb.ax.transAxes, horizontalalignment = 'right')
    chartBox = cb.ax.get_position()
    cb.ax.set_position([chartBox.x0+0.01, chartBox.y0-0.01, chartBox.width, chartBox.height])


    ax[1].plot(x, wt[:,-1])
    ax[1].set_xticks([0, l, 2*l])
    ax[1].set_xlabel(r'$x$', fontsize = fs)
    chartBox = ax[1].get_position()
    ax[1].set_position([chartBox.x0 + 0.115, chartBox.y0, chartBox.width*0.78, chartBox.height])
    ax[1].set_ylabel(f'{wlabel}', fontsize = fs)
    _, top = ax[1].get_ylim()
    ax[1].plot([l, l], [1, top], ':', color = 'gray')
    ax[1].set_ylim(bottom = 1, top = top)
    
    if ax_is_None:
        fig.tight_layout()
        return fig, ax
    
def iter_sweep(method, nT = 1, y0 = [0], w0 = 0.1, l = 8, d = 4, v = 3.2, k = 0.114*0.017, A = 0.025, rLTD = 0.3, r0 = 6, rx = 20, tau = 1, tau_m = 0.02, gL = 10, cap = 0, nt = 1000, nx = 0, tau_norm = 2.5, average_w = False, plot = False, theme = None, iter_method = 'RK45', quad = False, fork_ivp = True, shape = 'square', reverse = 0, func = None, iint_func = None, i_func = None, norm_w = 2, reset_fr = True, figsize = None, return_FG = False, F_perc = 1.0, dt = 0, diagram_perc = None, fs = 10, ls = 8, pdiagram_idx = -1, pd = [0.05, 0.27, 0.50, 0.72, 0.95], pscale = 2, less = 1, xunit = 'wave_front'):
    if nx == 0:
        nx = l*2 + 2

    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)
    if average_w and reverse > 0:
        print('reverse will not take effect when average_w is True')

    match method:
        case 'approx1'|'approx2'|'exact0'|'exact1'|'exact2':
            if len(y0) == 1: 
                y0 = np.repeat(y0, 2)
            elif len(y0) != 2:
                raise Exception(f'length of y0 should be 1 or 2 for {method}')
        case 'approx3':
            if len(y0) == 1: 
                y0 = np.repeat(y0, 3)
            elif len(y0) != 3:
                raise Exception(f'length of y0 should be 1 or 3 for {method}')
        case _:
            raise Exception(f'{method} not implemented')
                
    r0 /= rLTD
    k *= A
    buff_ratio = np.sqrt(2)
    L = l*2*np.sqrt(2) + 2*d
    T = L/v
    x = np.linspace(0,2*l,nx)
    if dt == 0:
        t = np.linspace(0,T,nt+1)
        dt = t[1] - t[0]
    else: #!!!! will result in inconsistent in get_xrange
        nt = int(T/dt)
        t = np.arange(nt+1)*dt
        T = t[-1]
        L = v*T
        buff_ratio = (L-2*d)/(2*l)
    t_total = np.arange(nT*nt+1)*dt
    w = w0*np.ones(nx)
    print(f'T = {T:.3f}, dt = {dt:.3e}, each LGN activated for {2*d/L*100:.1f}%')
    w0_max = w0*cap
    if cap > 0:
        print(f'max weight possible: {w0_max:.3e}')
    else:
        print(f'no cap on weight')

    match method:
        case 'approx1'|'approx2'|'approx3':
            wt = np.zeros((nx,nT+1))
        case 'exact0'|'exact1'|'exact2':
            wt = np.zeros((nx,nT*nt+1))

    print(f'using method {method}')
    wt[:,0] = w
    match method:
        case 'exact0'|'exact1'|'exact2':
            y0 = np.hstack((y0, w))
                                #  overlap 
    r = np.zeros(nT*nt+1) # |0  ... nt|0
    rbar = np.zeros(nT*nt+1)
    r[0] = y0[0]
    rbar[0] = y0[1]
    Gt = np.zeros(nT*nt+1)
    G0 = np.zeros(nT*nt+1)
    Ft = np.zeros(nT*nt+1)
    msg_len = 0
    if nT > 5:
        _nT = int(np.ceil(np.sqrt(nT)))
        r_nT = (nT + _nT - 1)//_nT
    else:
        _nT = nT
        r_nT = 1
    if plot:
        if figsize is None:
            fig = plt.figure(figsize = (_nT, r_nT*3.5), dpi = 200)
        else:
            fig = plt.figure(figsize = figsize, dpi = 200)

    if diagram_perc is not None:

        diagrami = np.array([int(round(pi*nT)) for pi in diagram_perc], dtype = int)
        diagrami[diagrami == 0] = 1
        diagrami = np.unique(diagrami)
        ndiagram = len(diagrami)

        if pdiagram_idx == -1:
            pdiagram_idx = np.arange(ndiagram)
        else:
            if not hasattr(pdiagram_idx, '__len__'):
                pdiagram_idx = np.array([pdiagram_idx])

        dheight = 2*len(pdiagram_idx)
        if less:
            nrows = 2
            if less == 2:
                height = dheight + 2.2
                wspace = 0.4
            else:
                height = dheight + 2.5
                wspace = 0.2
        else:
            nrows = 3
            height = dheight + 3.2
            wspace = 0.4

        width = ndiagram*1.9

        dfig = plt.figure(figsize = (width, height), dpi = 200)
        if less:
            if less == 2:
                height_ratios = [1.125, 1.075]
            else:
                height_ratios = [1.0, 0.8]
        else:
            height_ratios = [1.1, 1.0, 0.8]
        
        grids = gs.GridSpec(ncols = ndiagram, nrows = nrows, figure = dfig, wspace = wspace, height_ratios = height_ratios)

        dgrids = gs.GridSpec(ncols = len(pd), nrows = len(pdiagram_idx), figure = dfig)

        if less == 2:
            grids.update(bottom = dheight*1.1/height)
            dgrids.update(top = dheight*0.85/height)
        else:
            grids.update(bottom = dheight*1.1/height)
            dgrids.update(top = dheight*0.8/height)

        
    match shape:
        case 'circle':
            half_func = half_circle_height
            iint_func = iint_circle_chord
            i_func = int_circle_chord
            if quad:
                func = half_circle_height
            else:
                func = iint_circle_chord
        case 'square':
            half_func = half_square
            iint_func = iint_square
            i_func = int_square
            if quad:
                func = half_square
            else:
                func = iint_square
        case _:
            print('using customized shape')

    Gt[0], bot, top = deltaW_Area(r[0], rbar[0], iint_func, 0,k,l,d,v,r0,rx)
    G0[0]           = deltaW_Area(r[0], rbar[0], iint_func, 0,k,l,d,v,r0,rx,ret_G0 = True)
    Ft[0] = deltaArea_W(half_func, wt[:,0], False, rx, v, l, bot = bot, top = top)
    _G = k*rx*rx*(iint_func(l+d/2,l) - iint_func(l-d/2,l))/gL
    max_r0_G = _G*r0
    if max_r0_G < 1:
        print(f'r0 * G <= {max_r0_G} < 1, the larger fixed point of the linearized system is always stable')
    else:
        sqrt_r0_G = np.sqrt(r0*r0 - r0/_G)
        print(f'r of fixed points of the linearized system need to be outside [{r0 - sqrt_r0_G}, {r0 + sqrt_r0_G}] to be stable, the fixed points (transcritical) are separated at {r0*2/3}')

    OdeSolver.__init__ = new_init
    OdeSolver.step = new_step
    try:
        if reverse > 0:
            ir = 0
        flipped = False
        di = 0
        for i in range(nT):
            sys.stdout.write(f"\r{' '*msg_len}")
            msg = f'\r{i+1}/{nT}...'
            msg_len = len(msg)
            sys.stdout.write(msg)
            try:
                match method:
                    case 'approx1':

                        if fork_ivp:
                            sol = solve_ivp(get_approx1(w, i==0 or average_w, func, l = l, d = d, v = v, rx = rx, gL = gL, tau_m = tau_m, tau = tau, quad = quad), t, y0)
                        else:
                            sol = sp.integrate.solve_ivp(get_approx1(w, i==0 or average_w, func, dt = dt, l = l, d = d, v = v, rx = rx, gL = gL, tau_m = tau_m, tau = tau, quad = quad), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                    case 'approx2':
                        if fork_ivp:
                            sol = solve_ivp(get_approx2(w, i==0 or average_w, half_func, iint_func, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau = tau, F_perc = F_perc), t, y0)
                        else:
                            sol = sp.integrate.solve_ivp(get_approx2(w, i==0 or average_w, half_func, iint_func, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau = tau, F_perc = F_perc), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                    case 'approx3':
                        if fork_ivp:
                            sol = solve_ivp(get_approx3(w, i==0 or average_w, half_func, iint_func, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau_m = tau_m, tau = tau, print_out = False, F_perc = F_perc), t, y0)
                        else:
                            sol = sp.integrate.solve_ivp(get_approx3(w, i==0 or average_w, half_func, iint_func, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau_m = tau_m, tau = tau, print_out = False, F_perc = F_perc), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)
                        
                    case 'exact0':
                        if fork_ivp:
                            sol = solve_ivp(get_exact0(func, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau_m = tau_m, tau = tau, quad = quad), t, y0, w0_max)
                        else:
                            sol = sp.integrate.solve_ivp(get_exact0(func, dt = dt, cap = w0_max, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau_m = tau_m, tau = tau, quad = quad), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                    case 'exact1':
                        if fork_ivp:
                            sol = solve_ivp(get_exact1(func, dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gain = 1/gL, tau = tau, quad = quad), t, y0, w0_max)
                        else:
                            sol = sp.integrate.solve_ivp(get_exact1(func, dt, cap = w0_max, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gain = 1/gL, tau = tau, quad = quad), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                    case 'exact2':
                        if fork_ivp:
                            sol = solve_ivp(get_exact2(half_func, func, dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gain = 1/gL, tau = tau, quad = quad, F_perc = F_perc), t, y0, w0_max)
                        else:
                            sol = sp.integrate.solve_ivp(get_exact2(half_func, func, dt, cap = w0_max, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gain = 1/gL, tau = tau, quad = quad, F_perc = F_perc), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                    case _:
                        raise Exception(f'{method} not implemented')

                islice = slice(i*nt+1, i*nt+1+nt)
                r[islice] = sol.y[0,1:]
                rbar[islice] = sol.y[1,1:]
            except:
                if not plot:
                    fig = plt.figure(figsize = (_nT, r_nT*3.5), dpi = 200)
                for j in range(r_nT):
                    ax = fig.add_subplot(3*r_nT, 1, 3*j+1)
                    ax2 = ax.twinx()
                    if i-j*_nT > _nT:
                        sl = np.s_[j*_nT*nt:(j+1)*_nT*nt+1]
                    else:
                        sl = np.s_[j*_nT*nt:i*nt+1]
                    ax.plot(t_total[sl], r[sl], 'k', lw = 1, label = 'fr')
                    ax.plot(t_total[sl], rbar[sl], ':k', lw = 1, label = 'avg. fr')
                    ax.plot(t_total[sl], np.power(rbar[sl],2)/r0, ':r', lw = 1, alpha = 1, label = 'thres. fr')
                    ax.legend(fontsize = 'xx-small')
                    ax2.plot(t_total[sl], k*rx*rx*i_func(t_total[sl], v, d, l), ':g', lw = 1, alpha = 0.8, label = 'F(t)')
                    ax3 = fig.add_subplot(3*r_nT, 1, 3*j+2)
                    if i-j*_nT > _nT:
                        ax3.plot(t_total[sl], Gt[sl]/gL, ':r', lw = 1, alpha = 0.8, label = 'dw/dt*G(t)')
                        ax3.plot(t_total[sl], Ft[sl]/gL, ':b', lw = 1, alpha = 0.8, label = 'F(t)')
                    else:
                        for _i in range(j*_nT, i):
                            _sl = np.s_[_i*nt:(_i+1)*nt+1]
                            ax3.plot(t_total[_sl], Gt[_sl]/gL, ':r', lw = 1, alpha = 0.8, label = 'dw/dt*G(t)')
                            ax3.plot(t_total[_sl], Ft[_sl]/gL, ':b', lw = 1, alpha = 0.8, label = 'F(t)')
                        _sl = np.s_[i*nt:(i+1)*nt]
                        ax3.plot(t_total[_sl], deltaW_Area(r[_sl], rbar[_sl], iint_func, t_total[_sl],k,l,d,v,r0,rx, deltaOnly = True)/gL, ':r', lw = 1, alpha = 0.8, label = 'dw/dt*G(t)')
                        match method:
                            case 'approx1'|'approx2'|'approx3':
                                ax3.plot(t_total[_sl], deltaArea_W(half_func, wt[:,i], False,rx,v,l,t_total[_sl],d)/gL, ':b', lw = 1, alpha = 0.8, label = 'G(t)')
                            case 'exact0'|'exact1'|'exact2':
                                ax3.plot(t_total[_sl], [deltaArea_W(half_func,wt[:,i*nt+_i], False,rx,v,l,t_total[i*nt+_i],d)/gL for _i in range(nt)], ':b', lw = 1, alpha = 0.8, label = 'F(t) est.')

                    ax3.legend(fontsize = 'xx-small')
                if 'sol' in locals(): 
                    _fig, _ax = plt.subplots(figsize = (3, 3))
                    _ax.plot(sol.t, sol.y[0,:])
                    _ax.plot(sol.t, sol.y[1,:])
                raise Exception(f'sweep #{i} diverged')

            y0 = sol.y[:,-1]

            Gt[islice], bot, top = deltaW_Area(r[islice], rbar[islice], iint_func, t_total[islice], k, l, d, v, r0, rx)
            G0[islice] = deltaW_Area(r[islice], rbar[islice], iint_func, t_total[islice], k, l, d, v, r0, rx, ret_G0 = True)
            match method:
                case 'approx1'|'approx2'|'approx3':
                    if flipped: # update w according to flipped status
                        w = next_W(sol.t, np.flip(wt[:,i]), sol.y[0,:], sol.y[1,:], k,l,d,v,rx,r0)
                    else:
                        w = next_W(sol.t, wt[:,i], sol.y[0,:], sol.y[1,:], k,l,d,v,rx,r0)
                        
                    if reverse > 0 and ((ir + 1) % reverse) == 0: # reverse every ${reverse} sweeps
                        w = np.flip(w)
                        if not flipped:
                            flipped = True
                        else:
                            flipped = False

                    if norm_w > 0:
                        w_ = np.sum(wt[:,i])
                        dw = (w_ - w0*nx) * np.exp(-T/tau_m)
                        if norm_w == 1:
                            w *= (w0*nx + dw)/w_
                        elif norm_w == 2:
                            w += (w0*nx + dw - w_)/nx
                        
                    if cap > 0:
                        w[w > w0_max] = w0_max
                    w[w<0] = 0

                    if flipped: # save wt in original direction
                        wt[:,i+1] = np.flip(w)
                    else:
                        wt[:,i+1] = w
                        
                    if plot:
                        wax = fig.add_subplot(3*r_nT, _nT, (3*(i//_nT)+2)*_nT + (i%_nT+1))
                        if flipped: # draw w in original direction
                            wax.plot(x, np.flip(w))
                            wax.set_title(r'$\rightarrow$', fontsize = 'xx-small')
                        else:
                            wax.plot(x, w)
                            wax.set_title(r'$\leftarrow$', fontsize = 'xx-small')
                        wax.tick_params(axis='both', labelsize = 'xx-small')
                        if i%_nT == 0:
                            wax.set_ylabel('weight', fontsize = 'xx-small')
                            wax.set_xlabel('x (LGN)', fontsize = 'xx-small')
                        wax.set_ylim(0)

                    if less != 2 and diagram_perc is not None and di < ndiagram and diagrami[di] == i+1:
                        print(f'plot weight at sweep {i}')
                        wax = dfig.add_subplot(grids[1, di])
                        _w = w.copy()
                        if flipped: # draw w in original direction
                            wax.plot(x, np.flip(_w), '-k')
                            wax.set_title(r'$\rightarrow$', fontsize = fs)
                        else:
                            wax.plot(x, _w, '-k')
                            wax.set_title(r'$\leftarrow$', fontsize = fs)
                        wax.plot(x, wt[:,i], '--m')
                        wax.tick_params(axis='both', labelsize = ls)
                        if di == 0:
                            wax.set_ylabel('weight', fontsize = fs)
                        if di == ndiagram//2:
                            wax.set_xlabel('x (LGN)', fontsize = fs)
                        di += 1
                        
                    if average_w: # reverse does not take effect
                        w = np.average(w, weights = half_func(x,l))

                    Ft[islice] = deltaArea_W(half_func, wt[:,i], False, rx, v, l, bot = bot, top = top)

                    if reset_fr:
                        y0[0] = 0
                        #y0[1] = 0

                case 'exact0'|'exact1'|'exact2':
                    w = sol.y[2:,1:].copy()

                    if plot:
                        wax = fig.add_subplot(3*r_nT, _nT, (3*(i//_nT)+2)*_nT + (i%_nT+1))
                        if not flipped: # draw in original direction
                            wax.plot(x, w[:,-1])
                            wax.set_title(r'$\rightarrow$', fontsize = 'xx-small')
                        else:
                            wax.plot(x, np.flip(w[:,-1]))
                            wax.set_title(r'$\leftarrow$', fontsize = 'xx-small')
                        wax.tick_params(axis='both', labelsize = 'xx-small')
                        if i%_nT == 0:
                            wax.set_ylabel('weight', fontsize = 'xx-small')
                            wax.set_xlabel('x (LGN)', fontsize = 'xx-small')
                        wax.set_ylim(0)

                    if less != 2 and diagram_perc is not None and di < ndiagram and diagrami[di] == i+1:
                        print(f'plot weight at sweep {i}')
                        wax = dfig.add_subplot(grids[1, di])
                        if not flipped: # draw w in original direction
                            wax.plot(x, w[:,-1], '-k')
                            wax.set_title(r'$\rightarrow$', fontsize = fs)
                        else:
                            wax.plot(x, np.flip(w[:,-1]), '-k')
                            wax.set_title(r'$\leftarrow$', fontsize = fs)
                        wax.plot(x, wt[:,i*nt+1], '--m')
                        wax.tick_params(axis='both', labelsize = ls)
                        if di == 0:
                            wax.set_ylabel('weight', fontsize = fs)
                        if di == ndiagram//2:
                            wax.set_xlabel('x (LGN)', fontsize = fs)
                        wax.set_ylim(0)
                        di += 1

                    if norm_w > 0:
                        w_ = np.sum(w[:,0])
                        dw = (w_ - w0*nx) * np.exp(-T/tau_m)
                        if norm_w == 1:
                            w[:, -1] *= (w0*nx + dw)/w_
                        elif norm_w == 2:
                            w[:, -1] += (w0*nx + dw - w_)/nx
                    if cap > 0:
                        w[w[:,-1] > w0_max, -1] = w0_max
                    w[w[:,-1]<0,-1] = 0

                    y0[2:] = w[:,-1] # norm for next wave's initial value

                    if not flipped: # draw in original direction
                        wt[:,islice] = w
                    else:
                        wt[:,islice] = np.flip(w, axis = 0)

                    Ft[islice] = deltaArea_W(half_func, wt[:,islice], False, rx, v, l, bot = bot, top = top)

                    if reverse > 0 and ((ir + 1) % reverse) == 0: # reverse every ${reverse} sweeps
                        y0[2:] = np.flip(y0[2:])
                        if flipped == False:
                            flipped = True
                        else:
                            flipped = False
                    if reset_fr:
                        y0[0] = 0
                        #y0[1] = 0

            if reverse > 0:
                ir += 1
    except: 
        OdeSolver.__init__ = old_init
        OdeSolver.step = old_step
        raise

    OdeSolver.__init__ = old_init
    OdeSolver.step = old_step
        
    if plot:
        for j in range(r_nT):
            ax = fig.add_subplot(3*r_nT, 1, 3*j+1)
            ax2 = ax.twinx()
            if nT-j*_nT > _nT:
                sl = np.s_[j*_nT*nt:(j+1)*_nT*nt+1]
                shrink_ax = False
            else:
                sl = np.s_[j*_nT*nt:]
                if nT-j*_nT < _nT:
                    shrink_ax = True
                else:
                    shrink_ax = False
            ax.plot(t_total[sl], r[sl], 'k', label = 'fr')
            ax.plot(t_total[sl], rbar[sl], ':k', label = 'avg. fr')
            ax.plot(t_total[sl], np.power(rbar[sl],2)/r0, ':r', lw = 1.2, alpha = 1.0, label = 'thres. fr')
            if j == 0:
                ax.legend(fontsize = 'xx-small')
                ax.set_title(f'w0 = {w0}, d = {d}, l = {l} from {method}', fontsize = 'x-small')
            ax.set_ylabel('rate (Hz)', fontsize = 'xx-small')
            ax.tick_params(axis='both', labelsize = 'xx-small')
            ax2.plot(t_total[sl], G0[sl]/gL, ':g', lw = 0.8, alpha = 0.8)
            ax2.set_ylabel('G(t)', c = 'g', fontsize = 'xx-small')
            ax2.tick_params(axis='y', colors='g', labelsize = 'xx-small')
            ax2.spines['right'].set_color('g')
            ax3 = fig.add_subplot(3*r_nT, 1, 3*j+2)
            ax4 =ax3.twinx() 
            ax3.plot(t_total[sl], Gt[sl]/gL, ':r', lw = 1, alpha = 0.8, label = 'dw/dt*G(t)')
            ax3.set_ylabel('G(t)dw/dt (Hz/t)', c = 'r', fontsize = 'xx-small')
            ax3.spines['left'].set_color('r')
            ax3.tick_params(axis='x', labelsize = 'xx-small')
            ax3.set_xlabel('t (s)', fontsize = 'xx-small')
            ax3.tick_params(axis='y', colors='r', labelsize = 'xx-small')
            ax4.plot(t_total[sl], Ft[sl]/gL, ':b', lw = 1, alpha = 0.8, label = 'F(t)')
            ax4.set_ylabel('F(t) (Hz/t)', c = 'b', fontsize = 'xx-small')
            ax4.spines['right'].set_color('b')
            ax4.tick_params(axis='y', colors='b', labelsize = 'xx-small')
            if shrink_ax:
                pos = ax.get_position()
                _l = (pos.x1 - pos.x0)*(nT-j*_nT)/_nT
                pos.x1 = pos.x0 + _l
                ax.set_position(pos)
                pos = ax2.get_position()
                _l = (pos.x1 - pos.x0)*(nT-j*_nT)/_nT
                pos.x1 = pos.x0 + _l
                ax2.set_position(pos)
                pos = ax3.get_position()
                _l = (pos.x1 - pos.x0)*(nT-j*_nT)/_nT
                pos.x1 = pos.x0 + _l
                ax3.set_position(pos)
                pos = ax4.get_position()
                _l = (pos.x1 - pos.x0)*(nT-j*_nT)/_nT
                pos.x1 = pos.x0 + _l
                ax4.set_position(pos)

        fig.tight_layout()
        if theme is not None:
            fig.savefig(f'{theme}-{method}-{nT}.png')
            fig.savefig(f'{theme}-{method}-{nT}.svg')

    if diagram_perc is not None:
        axes = dfig.get_axes()
        match xunit:
            case 'time':
                xf = lambda x: x # time
                xlabel = 'time (s)'
            case 'wave_front': 
                xf = lambda x:x*v-(np.sqrt(2)-1)*l # wave front
                xlabel = r'wave front $\tilde{x}$'
            case 'rel_wave_front':
                xf = lambda x: x*v - l*np.sqrt(2) + d # relative wave front
                xlabel = r'rel. wave front $\tilde{x}_r$'
        i = 0
        di = 0
        for j in diagrami-1:
            ax = dfig.add_subplot(grids[0, i])
            p = (j+1)/nT
            sl = np.s_[j*nt:(j+1)*nt+1]
            print(f'plot sample sweep #{j}')
            t = t_total[sl] - t_total[j*nt]
            match method:
                case 'approx1'|'approx2'|'approx3':
                    temporal_trace(r0, G0[sl]/gL, Ft[sl]/gL, r[sl], rbar[sl], t, wt[:,j], False, half_func, iint_func, rx, v, l, k, d, gL, fs = fs, ax = ax, xf = xf, pscale = pscale, plot_ylabel = i == 0, plot_xlabel = i == ndiagram//2, pLeg = i == 0, less = less) 
                case 'exact0'|'exact1'|'exact2':
                    temporal_trace(r0, G0[sl]/gL, Ft[sl]/gL, r[sl], rbar[sl], t, wt[:,sl], False, half_func, iint_func, rx, v, l, k, d, gL, fs = fs, ax = ax, xf = xf, pscale = pscale, plot_ylabel = i == ndiagram//2, pLeg = i==0, less = less) 

            if less == 2 and i in pdiagram_idx:
                ax.plot(xf(T*np.array(pd)), np.interp(T*np.array(pd), t, r[sl]), ls = 'None', marker = '.', c = 'k', ms = 6, alpha = 1, zorder = 2)
            ax.set_title(rf'#{j+1} ({p*100:.0f}%)', fontsize = fs + 2)

            if i == 0:
                ax.set_ylabel('rate (Hz)', fontsize = fs)
            if less == 1 and i == ndiagram//2:
                ax.set_xlabel(xlabel, fontsize = fs)
            loc = plticker.MultipleLocator(base=l)
            ax.xaxis.set_major_locator(loc)
            ax.tick_params(axis='both', labelsize = fs-1)
            if less == 2:
                ax.tick_params(labelbottom = False)

            if less != 1:
                ax2 = ax.twinx()
                ax2.plot(xf(t), G0[sl]/gL, ':g', alpha = 0.8)
                ax2.set_ylim(top = ax2.get_ylim()[1]*1.2)
                if i == ndiagram-1:
                    ax2.set_ylabel('G(t)', c = 'g', fontsize = fs)
                ax2.tick_params(axis='y', colors='g', labelsize = fs-1)
                ax2.spines['right'].set_color('g')
                ax2.tick_params(labelbottom = False)
            if less == 1:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height*0.3, box.width, box.height*0.7])

            if less == 2:
                ax3 = dfig.add_subplot(grids[1, i])
                if i == ndiagram//2:
                    ax3.set_xlabel(xlabel, fontsize = fs)
                ax3.plot(xf(t), Ft[sl]/gL, '-b', alpha = 1.0, label = 'F(t)', zorder = 0)
                #ax3.plot(xf(T*np.array(pd)), np.interp(T*np.array(pd), t, Ft[sl]/gL), ls = 'None', marker = '*', c = 'b', ms = 6, alpha = 0.8) 
                if i in pdiagram_idx:
                    ax3.plot(xf(T*np.array(pd)), np.interp(T*np.array(pd), t, Ft[sl]/gL), ls = 'None', marker = '.', c = 'b', ms = 6, alpha = 1, zorder = 2)
                if i == 0:
                    ax3.set_ylabel('F(t)', c = 'b', fontsize = fs)
                ax3.spines['right'].set_color('g')
                ax3.spines['left'].set_color('b')
                ax3.tick_params(axis='y', colors='b', labelsize = fs-1)

                ax4 =ax3.twinx() 
                #ax4.plot(xf(t), Gt[sl]/gL, '-g', alpha = 1.0, label = r'$r(r-\bar{r}^2/r_{0})$ G(t)')
                ax4.plot(xf(t), Gt[sl]/gL, '-g', alpha = 1.0, label = r'$r(r-r_{th})$ G(t)')
                if i in pdiagram_idx:
                    ax4.plot(xf(T*np.array(pd)), np.interp(T*np.array(pd), t, Gt[sl]/gL), ls = 'None', marker = '.', c = 'g', ms = 6, alpha = 1, zorder = 2)
                if i == ndiagram-1:
                    #ax4.set_ylabel(r'$r(r-\bar{r}^2/r_{0})$ G(t)', c = 'g', fontsize = fs)
                    ax4.set_ylabel(r'$r(r-r_{th})$ G(t)', c = 'g', fontsize = fs)
                ax4.tick_params(axis='y', colors='g', labelsize = fs-1)
                #ax4.spines['right'].set_color('g')
                ax4.set_ylim(bottom = -ax4.get_ylim()[1])
                ax4.tick_params(axis='x', labelsize = fs-1)
                
            if less != 2:
                axes[i].set_ylim(bottom = max(0, w0 - (w0-wt.min())*1.2), top = w0 + (wt.max()-w0)*1.2)
                axes[i].tick_params(labelsize = fs-1)
                box = axes[i].get_position()
                axes[i].set_position([box.x0, box. y0, box.width, box.height*0.7])

            if less == 2 or less == 0:
                if i in pdiagram_idx:
                    pi = 0
                    for perc in pd:
                        ax = dfig.add_subplot(dgrids[di, pi])
                        match method:
                            case 'approx1'|'approx2'|'approx3':
                                #phase_portrait(ax, perc, rx, r0, 1, k, v, d, l, gL, tau, half_func, iint_func, i_func, T, wt[:,j], r[sl], rbar[sl], t, pi == len(pd)//2, pi == 0, scale = pscale, G = np.interp(T*perc, t, G0[sl]/gL), Ft = np.interp(T*perc, t, Ft[sl]/gL))
                                phase_portrait(ax, perc, rx, r0, 1, k, v, d, l, gL, tau, half_func, iint_func, i_func, T, wt[:,j], r[sl], rbar[sl], t, pi == len(pd)//2, pi == 0, scale = pscale, label_FS = fs-1, label_fs = fs-1)
                            case 'exact0'|'exact1'|'exact2':
                                phase_portrait(ax, perc, rx, r0, 1, k, v, d, l, gL, tau, half_func, iint_func, i_func, T, wt[:,sl], r[sl], rbar[sl], t, pi == len(pd)//2, pi == 0, scale = pscale, label_FS = fs-1, label_fs = fs-1)
                        pi += 1
                    di += 1
            i += 1

        #dfig.tight_layout()
        if theme is not None:
            dfig.savefig(f'{theme}-{method}-{nT}_summary.png')
            dfig.savefig(f'{theme}-{method}-{nT}_summary.svg')

    if return_FG:
        return wt, r, rbar, dt, Gt/gL, G0/gL, Ft/gL
    else:
        return wt, r, rbar

def get_dF2_III(w_on, w_off, uniform_w, half_fxt, iint_fxt, t, k = 0.114*0.017*0.025, r0 = 6, l = 8, d = 3, v = 3.2, rx = 20, gL = 40, tau = 1):
    def dF(r, r_bar):
        return dInput_approx_III(r, r_bar, w_on, w_off, uniform_w, half_fxt, iint_fxt, t, k = k, l = l, d = d, v = v, r0 = r0, rx = rx)/gL, dRbar(r_bar, r, tau = tau)
    return dF

def dR_approx2_III(r, r_bar, w_on, w_off, uniform_w, half_fxt, iint_fxt, t, dt = None, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gL = 40):
    dI = dInput_approx_III(r, r_bar, w_on, w_off, uniform_w, half_fxt, iint_fxt, t, k = k, l = l, d = d, v = v, r0 = r0, rx = rx)
    if dt is not None:
        pick = r + dI*dt < 0
        dI[pick] = -r[pick]/dt
    return dI/gL

def nullcline2_III(r, w_on, w_off, uniform_w, half_fxt, iint_fxt, t, k, l, d, v, r0, rx, gL):
    bot_on, top_on = get_xrange(t,v,d,l,1)
    Fx_on = iint_fxt(top_on,l) - iint_fxt(bot_on,l)
    bot_off, top_off = get_xrange(t-d/v,v,d,l,1)
    Fx_off = iint_fxt(top_off,l) - iint_fxt(bot_off,l)
    Fx = Fx_on + Fx_off
    #print(bot,top,Fx)
    if Fx == 0:
        if hasattr(r, '__len__'):
            rbar = np.zeros_like(r)
            rbar[:] = np.nan
            return rbar, np.nan
        else:
            return np.nan, np.nan
        
    if uniform_w:
        if hasattr(w_on, '__len__'):
            fxw_on = 2*(half_fxt(top_on, l) - half_fxt(bot_on, l))*w_on[0]
            fxw_off = 2*(half_fxt(top_off, l) - half_fxt(bot_off, l))*w_off[0]
        else:
            fxw_on = 2*(half_fxt(top_on, l) - half_fxt(bot_on, l))*w_on
            fxw_off = 2*(half_fxt(top_off, l) - half_fxt(bot_off, l))*w_off

    else:
        nx = len(w_on)
        x = np.linspace(0,2*l,nx)
        w_top = np.interp(top_on, x, w_on)
        w_bot = np.interp(bot_on, x, w_on)
        fxw_on = 2*(half_fxt(top_on, l)*w_top - half_fxt(bot_on, l)*w_bot)
        w_top = np.interp(top_off, x, w_off)
        w_bot = np.interp(bot_off, x, w_off)
        fxw_off = 2*(half_fxt(top_off, l)*w_top - half_fxt(bot_off, l)*w_bot)

    fxw = fxw_on + fxw_off

    if hasattr(r, '__len__'):
        rbar = np.zeros_like(r)
        _tmp = np.zeros_like(r)
        rpick = r!=0
        _tmp[rpick] = v*fxw/(k*rx*Fx*r[rpick]) + r[rpick]
        #print(_tmp*r0)
        unpick = np.logical_or(r == 0, _tmp < 0)
        if sum(unpick) > 0:
            rbar[unpick] = np.nan
        pick = np.logical_not(unpick)
        if sum(pick) > 0:
            rbar[pick] = np.sqrt(_tmp[pick]*r0)
        if fxw > 0:
            ro = np.nan
        else:
            ro = np.sqrt(-v*fxw/(k*rx*Fx))
        return rbar, ro
    else:
        if r == 0:
            return np.nan
        else:
            _tmp = v*fxw/(k*rx*Fx*r) + r
            if _tmp >= 0:
                return np.sqrt(_tmp*r0)
            else:
                return np.nan

def R_III(w, nx, uniform_w, iint_fxt, t, gain, l = 8, d = 3, v = 3.2, rx = 20, quad = False):
    return gain*rx*(int_fxt_and_W(w[:nx,:], uniform_w, iint_fxt, t, l=l, d=d, v=v, nd=1, quad=quad) + int_fxt_and_W(w[nx:,:], uniform_w, iint_fxt, t-d/v, l=l, d=d, v=v, nd=1, quad=quad))

def get_exact1_III(iint_fxt, dt, cap = 0, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20, gain = 0.1, tau = 1, skip_LGN = None, quad = False):
    def func(t, y):
        nx = (y.shape[0]-2)//2
        assert(np.mod(y.shape[0]-2, 2) == 0)
        y_next = np.zeros_like(y)
        r = R_III(y[2:,:], nx, False, iint_fxt, t, gain, l=l, d=d, v=v, rx=rx, quad = quad)
        y_next[0,:] = (r - y[0,:])/dt
        y_next[1,:] = dRbar(y[1,:], r, tau = tau)
        if skip_LGN is not None:
            y_next[2:2+nx,:] = dW(r, y[1,:], k,l,d,v,r0,rx,t,nx,dt,y[2:2+nx,:], nd = 1, cap = cap, skip_LGN = skip_LGN[0,:])
            y_next[2+nx:2+2*nx,:] = dW(r, y[1,:], k,l,d,v,r0,rx,t-d/v,nx,dt,y[2+nx:2+2*nx,:], nd = 1, cap = cap, skip_LGN = skip_LGN[1,:])
        else:
            y_next[2:2+nx,:] = dW(r, y[1,:], k,l,d,v,r0,rx,t,nx,dt,y[2:2+nx,:], nd = 1, cap = cap)
            y_next[2+nx:2+2*nx,:] = dW(r, y[1,:], k,l,d,v,r0,rx,t-d/v,nx,dt,y[2+nx:2+2*nx,:], nd = 1, cap = cap)
        return y_next
    return func

def get_approx2_III(w_on, w_off, uniform_w, half_fxt, iint_fxt, dt = None, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 1, gL = 40, tau = 1):
    def func(t, y):
        y_next = np.zeros_like(y)
        y_next[0,:] = dR_approx2_III(y[0,:], y[1,:], w_on, w_off, uniform_w, half_fxt, iint_fxt, t, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL) # R'
        y_next[1,:] = dRbar(y[1,:], y[0,:], tau = tau) # R_bar'
        return y_next
    return func

def dInput_approx_III(r, r_bar, w_on, w_off, uniform_w, half_fxt, iint_fxt, t, k = 0.114*0.017*0.025, l = 8, d = 3, v = 3.2, r0 = 6, rx = 20):
    # approx: w fixed during wave
    # w_on
    bot, top = get_xrange(t,v,d,l,1)
    thres = (r_bar/r0)*r_bar
    Fx1 = k*rx*r*(r - thres)*(iint_fxt(top,l) - iint_fxt(bot,l))
    if uniform_w:
        if hasattr(w_on, '__len__'):
            Q_on = rx*(Fx1 + 2*v*(half_fxt(top, l) - half_fxt(bot, l))*w_on[0])
        else:
            Q_on = rx*(Fx1 + 2*v*(half_fxt(top, l) - half_fxt(bot, l))*w_on)
    else:
        nx = len(w_on)
        x = np.linspace(0,2*l,nx)
        w_top = np.interp(top, x, w_on)
        w_bot = np.interp(bot, x, w_on)
        Q_on = rx*(Fx1 + 2*v*(half_fxt(top, l)*w_top - half_fxt(bot, l)*w_bot))

    bot, top = get_xrange(t-d/v,v,d,l,1)
    thres = (r_bar/r0)*r_bar
    Fx1 = k*rx*r*(r - thres)*(iint_fxt(top,l) - iint_fxt(bot,l))
    if uniform_w:
        if hasattr(w_off, '__len__'):
            Q_off = rx*(Fx1 + 2*v*(half_fxt(top, l) - half_fxt(bot, l))*w_off[0])
        else:
            Q_off = rx*(Fx1 + 2*v*(half_fxt(top, l) - half_fxt(bot, l))*w_off)
    else:
        nx = len(w_off)
        x = np.linspace(0,2*l,nx)
        w_top = np.interp(top, x, w_off)
        w_bot = np.interp(bot, x, w_off)
        Q_off = rx*(Fx1 + 2*v*(half_fxt(top, l)*w_top - half_fxt(bot, l)*w_bot))

    return Q_on + Q_off

def iter_sweep_III(method, nT = 1, y0 = [0], w0 = 0.1, l = 8, d = 4, v = 3.2, k = 0.114*0.017, A = 0.025, rLTD = 0.3, r0 = 6, rx = 20, tau = 1, tau_m = 0.02, gL = 10, cap = 0, nt = 1000, nx = 0, average_w = False, plot = False, theme = None, iter_method = 'RK45', quad = False, fork_ivp = True, shape = 'square', reverse = 0, func = None, iint_func = None, i_func = None, skip_LGN = None):
    if theme is not None:
        plot = True
    if nx == 0:
        nx = l*2 + 2

    if not isinstance(y0, np.ndarray):
        y0 = np.array(y0)

    if average_w and reverse > 0:
        print('reverse will not take effect when average_w is True')

    match method:
        case 'approx2'|'exact1':
            if len(y0) == 1: 
                y0 = np.repeat(y0, 2)
            elif len(y0) != 2:
                raise Exception(f'length of y0 should be 1 or 2 for {method}')
        case _:
            raise Exceptionf('{method} not implemented')
                

    r0 /= rLTD
    k *= A
    L = l*2*np.sqrt(2) + 2*d
    T = L/v
    x = np.linspace(0,2*l,nx)
    t = np.linspace(0,T,nt+1)
    dt = t[1] - t[0]
    t_total = np.linspace(0,nT*T,nt*nT+1)
    w_on = w0*np.ones(nx)
    w_off = w0*np.ones(nx)
    print(f'T = {T:.3f}, dt = {dt:.3e}, each LGN activated for {2*d/L*100:.1f}%')
    w0_max = w0*cap
    if cap > 0:
        print(f'max weight possible: {w0_max:.3e}')
    else:
        print(f'no cap on weight')

    match method:
        case 'approx2':
            wt_on = np.zeros((nx,nT+1))
            wt_off = np.zeros((nx,nT+1))
        case 'exact1':
            wt_on = np.zeros((nx,nT*nt+1))
            wt_off = np.zeros((nx,nT*nt+1))
            
    if skip_LGN is not None and nx != 0:
        if skip_LGN.shape[0] != 2 or skip_LGN.shape[1] != nx:
            raise Exception('skip_LGN should have the shape of [2,nx]')
        w_on[skip_LGN[0,:]] = 0
        w_off[skip_LGN[1,:]] = 0
    else:
        skip_LGN = np.zeros((2,nx), dtype = bool)

    print(f'using method {method}')
    wt_on[:,0] = w_on
    wt_off[:,0] = w_off
    match method:
        case 'exact1':
            y0 = np.hstack((y0, w_on, w_off))
                                #  overlap 
    r = np.zeros(nT*nt+1) # |0  ... nt|0
    rbar = np.zeros(nT*nt+1)
    r[0] = y0[0]
    rbar[0] = y0[1]
    msg_len = 0
    if plot:
        _nT = int(np.ceil(np.sqrt(nT)))
        fig = plt.figure(figsize = (_nT, _nT*1.5), dpi = 120)

    match shape:
        case 'circle':
            half_func = half_circle_height
            iint_func = iint_circle_chord
            i_func = int_circle_chord
            if quad:
                func = half_circle_height
            else:
                func = iint_circle_chord
        case 'square':
            half_func = half_square
            iint_func = iint_square
            i_func = int_square
            if quad:
                func = half_square
            else:
                func = iint_square
        case _:
            print('using customized shape')

    OdeSolver.__init__ = new_init
    OdeSolver.step = new_step
    try:
        if reverse > 0:
            ir = 0
        flipped = False
        for i in range(nT):
            sys.stdout.write(f"\r{' '*msg_len}")
            msg = f'\r{i+1}/{nT}...'
            msg_len = len(msg)
            sys.stdout.write(msg)
            match method:
                case 'approx2':
                    if fork_ivp:
                        sol = solve_ivp(get_approx2_III(w_on, w_off, i==0 or average_w, half_func, iint_func, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau = tau), t, y0)
                    else:
                        sol = sp.integrate.solve_ivp(get_approx2_III(w_on, w_off, i==0 or average_w, half_func, iint_func, dt = dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gL = gL, tau = tau), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                case 'exact1':
                    if fork_ivp:
                        sol = solve_ivp(get_exact1_III(func, dt, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gain = 1/gL, tau = tau, skip_LGN = skip_LGN, quad = quad), t, y0, w0_max)
                    else:
                        sol = sp.integrate.solve_ivp(get_exact1_III(func, dt, cap = w0_max, k = k, l = l, d = d, v = v, r0 = r0, rx = rx, gain = 1/gL, tau = tau, skip_LGN = skip_LGN, quad = quad), [0, T], y0, vectorized = True, t_eval = t, method = iter_method)

                case _:
                    raise Exception(f'{method} not implemented')

            islice = slice(i*nt+1, i*nt+1+nt)
            try:
                r[islice] = sol.y[0,1:]
                rbar[islice] = sol.y[1,1:]
            except:
                if plot:
                    for j in range(_nT):
                        ax = fig.add_subplot(2*_nT, 1, 2*j+1)
                        ax2 = ax.twinx()
                        if nT-j*_nT > _nT:
                            sl = np.s_[j*_nT*nt:(j+1)*_nT*nt+1]
                        else:
                            sl = np.s_[j*_nT*nt:]
                        ax.plot(t_total[sl], r[sl], 'k', label = 'fr')
                        ax.plot(t_total[sl], rbar[sl], ':k', label = 'avg. fr')
                        ax.plot(t_total[sl], np.power(rbar[sl],2)/r0, ':r', lw = 1.2, alpha = 1.0, label = 'thres. fr')
                        ax.legend(fontsize = 'xx-small')
                        ax2.plot(t_total[sl], k*rx*rx*i_func(t_total[sl], v, d, l,1), ':m', lw = 0.8, alpha = 0.8, label = 'on F(t)')
                        ax2.plot(t_total[sl], k*rx*rx*i_func(t_total[sl]-d/v, v, d, l,1), ':c', lw = 0.8, alpha = 0.8, label = 'off F(t)')
                _fig, _ax = plt.subplots(figsize = (3, 3))
                _ax.plot(sol.t, sol.y[0,:])
                _ax.plot(sol.t, sol.y[1,:])
                raise Exception(f'sweep{i}, diverged')

            y0 = sol.y[:,-1]

            match method:
                case 'approx2':
                    if flipped: # update w according to flipped status
                        w_on = next_W(sol.t, np.flip(wt_on[:,i]), sol.y[0,:], sol.y[1,:], k,l,d,v,rx,r0, nd = 1, skip_LGN = np.flip(skip_LGN[0,:]))
                        w_off = next_W(sol.t-d/v, np.flip(wt_off[:,i]), sol.y[0,:], sol.y[1,:], k,l,d,v,rx,r0, nd = 1, skip_LGN = np.flip(skip_LGN[1,:]))
                    else:
                        w_on = next_W(sol.t, wt_on[:,i], sol.y[0,:], sol.y[1,:], k,l,d,v,rx,r0, nd = 1, skip_LGN = skip_LGN[0,:])
                        w_off = next_W(sol.t-d/v, wt_off[:,i], sol.y[0,:], sol.y[1,:], k,l,d,v,rx,r0, nd = 1, skip_LGN = skip_LGN[0,:])

                    if reverse > 0 and ((ir + 1) % reverse) == 0: # reverse every ${reverse} sweeps
                        w_on = np.flip(w_on)
                        w_off = np.flip(w_off)
                        if flipped == False:
                            flipped = True
                        else:
                            flipped = False

                    if cap > 0:
                        w_on[w_on > w0_max] = w0_max
                    if cap > 0:
                        w_off[w_off > w0_max] = w0_max
                    w_on[w_on < 0] = 0
                    w_off[w_off < 0] = 0

                    if flipped: # save wt in original direction
                        wt_on[:,i+1] = np.flip(w_on)
                        wt_off[:,i+1] = np.flip(w_off)
                    else:
                        wt_on[:,i+1] = w_on
                        wt_off[:,i+1] = w_off

                    if plot:
                        wax = fig.add_subplot(2*_nT, _nT, (2*(i//_nT)+1)*_nT + (i%_nT+1))
                        _w_on = w_on.copy()
                        _w_off = w_off.copy()
                        _w_on[w_on == 0] = np.nan
                        _w_off[w_off == 0] = np.nan
                        if flipped: 
                            wax.plot(x, np.flip(_w_on), 'r')
                            wax.plot(x, np.flip(_w_off), 'b')
                            wax.set_title(f'<-')
                        else:
                            wax.plot(x, _w_on, 'r')
                            wax.plot(x, _w_off, 'b')
                            wax.set_title(f'->')
                        wax.set_ylim(0)

                    if average_w: # reverse does not take effect
                        w_on = np.average(w_on, weights = func(x,l))
                        w_off = np.average(w_off, weights = func(x,l))

                case 'exact1':
                    islice = slice(i*nt+1, i*nt+1+nt)
                    w_on = sol.y[2:2+nx,-1].copy()
                    w_off = sol.y[2+nx:2+2*nx,-1].copy()
                    if not flipped: # note the code squence, wt should be updated "flipped" is set below, save wt in original direction
                        wt_on[:,islice] = sol.y[2:2+nx,1:]
                        wt_off[:,islice] = sol.y[2+nx:2+2*nx,1:]
                    else:
                        wt_on[:,islice] = np.flip(sol.y[2:2+nx,1:], axis = 0)
                        wt_off[:,islice] = np.flip(sol.y[2+nx:2+2*nx,1:], axis = 0)

                    if plot:
                        wax = fig.add_subplot(2*_nT, _nT, (2*(i//_nT)+1)*_nT + (i%_nT+1))
                        w_on[w_on == 0] = np.nan
                        w_off[w_off == 0] = np.nan
                        if not flipped: 
                            wax.plot(x, w_on, 'r')
                            wax.plot(x, w_off, 'b')
                            wax.set_title(f'->')
                        else:
                            wax.plot(x, np.flip(w_on), 'r')
                            wax.plot(x, np.flip(w_off), 'b')
                            wax.set_title(f'<-')
                        wax.set_ylim(0)

                    if reverse > 0 and ((ir + 1) % reverse) == 0: # reverse every ${reverse} sweeps
                        y0[2:2+nx] = np.flip(y0[2:2+nx])
                        y0[2+nx:2+2*nx] = np.flip(y0[2+nx:2+2*nx])
                        skip_LGN = np.flip(skip_LGN, axis = 1)
                        if flipped == False:
                            flipped = True
                        else:
                            flipped = False
                            
            if reverse > 0:
                ir += 1

    except: 
        OdeSolver.__init__ = old_init
        OdeSolver.step = old_step
        raise


    OdeSolver.__init__ = old_init
    OdeSolver.step = old_step
        
    if plot:
        for j in range(_nT):
            ax = fig.add_subplot(2*_nT, 1, 2*j+1)
            ax2 = ax.twinx()
            if nT-j*_nT > _nT:
                sl = np.s_[j*_nT*nt:(j+1)*_nT*nt+1]
            else:
                sl = np.s_[j*_nT*nt:]
            ax.plot(t_total[sl], r[sl], 'k', label = 'fr')
            ax.plot(t_total[sl], rbar[sl], ':k', label = 'avg. fr')
            ax.plot(t_total[sl], np.power(rbar[sl],2)/r0, ':r', lw = 1.2, alpha = 1.0, label = 'thres. fr')
            if j == 0:
                ax.legend(fontsize = 'xx-small')
            ax.set_title(f'w0 = {w0}, d = {d}, l = {l} from {method}')
            ax2.plot(t_total[sl], k*rx*rx*i_func(t_total[sl], v, d, l, 1), ':m', lw = 0.8, alpha = 0.8, label = 'on F(t)')
            ax2.plot(t_total[sl], k*rx*rx*i_func(t_total[sl]-v/d, v, d, l, 1), ':c', lw = 0.8, alpha = 0.8, label = 'off F(t)')
        fig.tight_layout()
        if theme is not None:
            fig.savefig(f'{theme}-{method}-{nT}.png')
            fig.savefig(f'{theme}-{method}-{nT}.svg')
    return wt_on, wt_off, r, rbar
