#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gs
import matplotlib as mpl
#import mpl_toolkits.mplot3d.art3d.Line3DCollection
import sys
from wave_approx import *
from wa_phase_diagram import *
mpl.rcParams.update({'font.family': 'CMU Sans Serif', 'axes.unicode_minus' : False})
mpl.rcParams.update({'mathtext.fontset': 'cm'})
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
"""
basic approx.:
    Poisson approximation of Triplet rule + rate homeostasis
    Wilson-Cowan V1
exact0: basic approx.
approx3: basic approx + update W only between sweeps (fixed within sweep)
approx2: approx3 + firing rate follows input instantaneously
approx1: approx2 + ignore W-r interaction within the sweep
""";


# In[2]:


quad = False
fork_ivp = True
square = False 

n = 5
p = np.linspace(0.05,0.95,n)
#p[3] = 0.48
#p[5] = 0.502
#p[6] = 0.52
l = 8
rLTD = 0.3
r0 = 6
rx = 20
tau = 1
d = 4
v = 3.2
A = 0.002
k = 0.114*0.017*A
print(k)
tau_m = 0.02
gL = 40
w0 = 0.1
cap = 32
dt = 1e-3

y0 = np.array([0, 0])
y1 = np.array([0, 0])
y2 = np.array([0, 0])
y3 = np.array([0, 0, 0])

L = 2*l*np.sqrt(2) + 2*d
T = L/v
print(f'T = {T:.3f}s')
# first T
nt = int(T/dt)+2
t = np.arange(nt)*dt
print(f'dt = {dt:.3e}, t[-1] = {t[-1]}')
nx = 101
x = np.linspace(0,2*l, nx)
w_init = w0*np.ones(nx)

if not square:
    half_form = half_circle_height
    iint_form = iint_circle_chord
    int_form = int_circle_chord
    if quad:
        form = half_circle_height
    else:
        form = iint_circle_chord
else:
    half_form = half_square 
    iint_form = iint_square
    int_form = int_square
    if quad:
        form = half_square
    else:
        form = iint_square
 


# In[3]:


r_max = 50
r = np.linspace(0, r_max, 30)
rbar = np.linspace(0, r_max, 30)
# solve approx2, plot trace
fig = plt.figure(figsize = (9,8), dpi = 150)
hr = [7,7,8,7]
grid = gs.GridSpec(ncols = 5, nrows = 4, figure = fig, height_ratios = hr, hspace = 0.4)

sol2 = solve_ivp(get_approx2(w0, True, half_form, iint_form, k = k, l = l, d = d, v = v, r0 = r0/rLTD, rx = rx, gL = gL, tau = tau), t, y2)

# phase plane
F = k*rx*rx*int_form(T*p, v=v, d=d, l=l)/gL
mks = {'Stable': '^m', 'Unstable': '*m', 'Saddle':'om', 'Degenerated': 'dm', 'Non-isolated':'sm'}
label_fs = 'medium'
label_FS = 'large'
for i in range(n):
    ax = fig.add_subplot(grid[2, i])
    phase_plane(get_dF2(w0, True, half_form, iint_form, T*p[i], k = k, r0 = r0/rLTD, l = l, d = d, v = v, rx = rx, tau = tau, gL = gL), r, rbar, ax = ax)
    _r = np.linspace(0,r[-1], 100)
    _rbar, ro = nullcline2(_r, w0, True, half_form, iint_form, T*p[i], k, l, d, v, r0/rLTD, rx, gL)
    ax.plot(_r, _rbar, 'g', lw = 2, alpha = 1, zorder = 0)
    _x_bot, _x_top = get_xrange(T*p[i],v,d,l)
    _G_sign = half_form(_x_top, l) - half_form(_x_bot, l)
    if _G_sign <= 1e-14:
        if np.isnan(ro):
            ro = 0
        iro = np.nonzero(ro - _r <= 0)[0][0]
        __r = np.insert(_r, max([0,iro]), ro)
        _rbar[iro:][np.isnan(_rbar[iro:])] = 0
        __rbar = np.insert(_rbar, max([0,iro]), 0)
        ax.plot(__r, __rbar, 'g', lw = 2, alpha = 1)
        #ax.plot(ro, 0, 'og', ms = 8, fillstyle = 'none', mew = 2, lw = 2, alpha = 1, zorder = 0)
    ax.plot(_r, _r, lw = 2, alpha = 1, zorder = 0)
    x_bot, x_top = get_xrange(T*p[i],v,d,l)
    Gt = 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL
    roots, types = solve_for_fp(r0/rLTD, F[i], Gt)
    if roots is not None:
        for root, typ in zip(roots, types):
            ax.plot(root, root, mks[typ], mfc = 'none', mew = 1.5)
        #ax.set_title(f'{p[i]*100:.0f}% T \n F(t) = {F[i]:.2e}\n G(t)={Gt:.1f}', fontsize = 'xx-small')
        ax.set_title(f'{p[i]*100:.0f}% T', fontsize = label_FS)
    else:
        #ax.set_title(f'{p[i]*100:.0f}% T \n F(t) = {F[i]:.2e}\n G(t)={Gt:.1f}, {types}', fontsize = 'xx-small')
        #ax.set_title(f'{p[i]*100:.0f}% T\n{types}', fontsize = 'xx-small')
        ax.set_title(f'{p[i]*100:.0f}% T', fontsize = label_FS)
    if i > 0:
        ax.set_yticklabels([])
    it = np.nonzero(T*p[i] - sol2.t <= 0)[0][0]
    ax.plot(sol2.y[0,it], sol2.y[1,it], '*k', ms = 6, alpha = 0.8)
    ax.set_xlim(r[0], r[-1])
    ax.tick_params(axis = 'both', labelsize = label_fs)
    ax.set_ylim(rbar[0], rbar[-1])
    ax.set_aspect('equal')
    if i == 0:
        ax.set_ylabel('avg. FR (Hz)')
    if i == n//2:
        ax.set_xlabel('FR (Hz)')

ax = fig.add_subplot(grid[0, 2:])
xt = sol2.t*v-2*l*(np.sqrt(2)-1)/2
ax.plot(xt, sol2.y[0,:], 'b', label = 'FR')
ax.plot(xt, sol2.y[1,:], 'g', label = 'avg. FR', alpha = 0.7)
print(f'k = {k:.3e}, coef_0 = {k*rx*rx:.3e}, coef_1 = {2*v*rx*w0}')
ax.plot(xt, np.power(sol2.y[1,:],2)/r0*rLTD, ':r', label = 'thres. FR')

ax.set_ylabel('rates (Hz)', fontsize = label_FS)
ax.legend(loc = 'upper left')
#ax.set_xlabel('wave front (#LGN = vt, v = 3.2 LGN/s)', fontsize = label_FS)
ax.set_xticklabels([])
ax.spines['right'].set_color('gray')
ax2 = ax.twinx()
ax2.plot(T*p*v-2*l*(np.sqrt(2)-1)/2, F, '*k', ms = 5, alpha = 0.8)
ax2.plot(t*v-2*l*(np.sqrt(2)-1)/2, k*rx*rx*int_form(t, v=v, d=d, l=l)/gL, ':k', lw = 2, alpha = 0.5, label = 'G(t)')
ax2.legend(labelcolor = 'gray')
ax2.set_ylabel('input gain G(t)', fontsize = label_FS, color = 'gray')
ax2.tick_params(axis = 'y', colors = 'gray')
ax2.spines['right'].set_color('gray')
chartBox = ax.get_position()
ax.set_position([chartBox.x0 - 0.05, chartBox.y0, chartBox.width*1, chartBox.height])
#ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.9, chartBox.height])

ax = fig.add_subplot(grid[1, 2:])
x_bot, x_top = get_xrange(t,v,d,l)
ax.plot(t*v-2*l*(np.sqrt(2)-1)/2, 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, color = 'k', alpha = 0.5)
_x_bot, _x_top = get_xrange(T*p,v,d,l)
ax.plot(T*p*v-2*l*(np.sqrt(2)-1)/2, 2*v*rx*w0*(half_form(_x_top, l) - half_form(_x_bot, l))/gL, '*k', ms = 6, alpha = 0.8)
ax.set_ylabel('flux F(t)', fontsize = label_FS)
#ax.set_xlabel('wave front (#LGN)', fontsize = label_FS)
ax.set_xlabel('wave front, x (#LGN)', fontsize = label_FS)
chartBox = ax.get_position()
ax.set_position([chartBox.x0 - 0.05, chartBox.y0+0.038, chartBox.width*1, chartBox.height])
#ax.set_position([chartBox.x0, chartBox.y0+0.038, chartBox.width*0.9, chartBox.height])

ax0 = fig.add_subplot(grid[3, 0])
ax1 = fig.add_subplot(grid[3, 1])
# plot W over time
w2 = next_W(t,w_init,sol2.y[0,:],sol2.y[1,:], k,l,d,v,rx,r0/rLTD, end_only = False)
w2[w2>w0*cap] = w0*cap
show_w_over_t(w2/w0, dt, 't (s)', 'w/w(0)', l, ax = [ax0, ax1], fs = label_FS)

fig.savefig('single_wave.png')
fig.savefig('single_wave.svg')


# In[4]:


### iter_sweep('exact2', nT = 1, w0 = w0, l = l, d = d, v = v, k = k/A, A = A, rLTD = rLTD, r0 = r0, rx = rx, tau = tau, tau_m = tau_m, gL = gL, cap = cap, nx = nx, nt = nt, plot = True, theme = 'check', fork_ivp = fork_ivp, shape = 'circle', reverse = 1, norm_w = 0, figsize = (10, 6));


# In[5]:


r_max =30 
r = np.linspace(-2, r_max, 20)
rbar = np.linspace(-2, r_max, 20)
perc = 0.51
F = k*rx*rx*int_form(T*perc, v=v, d=d, l=l)/gL
fig, ax = plt.subplots(dpi = 200)
phase_plane(get_dF2(w0, True, half_form, iint_form, T*perc, k = k, r0 = r0/rLTD, l = l, d = d, v = v, rx = rx, tau = tau, gL = gL), r, rbar, ax = ax, d = 4)
_r = np.linspace(0,r[-1], 1000)
_rbar, ro = nullcline2(_r, w0, True, half_form, iint_form, T*perc, k, l, d, v, r0/rLTD, rx, gL)
ax.plot(_r, _rbar, 'g', lw = 1, alpha = 1, zorder = 0)

if ro is not None:
    ax.plot(ro, 0, 'og', fillstyle = 'none', alpha = 1, zorder = 0)
ax.plot(_r, _r, lw = 1, alpha = 1, zorder = 0)
x_bot, x_top = get_xrange(T*perc,v,d,l)
Gt = 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL
roots, types = solve_for_fp(r0/rLTD, F, Gt)
mks = {'Stable': '*g', 'Unstable': '*r', 'Saddle':'gs', 'Non-isolated': 'g^'}
if roots is not None:
    for root, typ in zip(roots, types):
        ax.plot(root, root, mks[typ])
    ax.set_title(f'{perc*100:.0f}% T \n F(t) = {F:.2e}\n G(t)={Gt:.1f}', fontsize = 'xx-small')
else:
    ax.set_title(f'{perc*100:.0f}% T \n F(t) = {F:.2e}\n G(t)={Gt:.1f}, {types}', fontsize = 'xx-small')
ax.set_xlim(-2,r_max)
ax.set_ylim(-2,r_max)
ax.set_aspect('equal')


# In[6]:


def get_Gtau(which, f, r0_rLTD):
    if r0_rLTD*f >= 1:
        if which == 1:
            tau_r = r0_rLTD + np.sqrt(f*f*r0_rLTD*r0_rLTD - f*r0_rLTD)/f
        else:
            tau_r = r0_rLTD - np.sqrt(f*f*r0_rLTD*r0_rLTD - f*r0_rLTD)/f
        if tau_r*f*(3*tau_r/r0_rLTD-2) <= 0:
            return np.nan
        else:
            return np.power(tau_r,2)*(tau_r - r0_rLTD)/r0_rLTD*f
    else:
        return np.nan

def stability_diagram(Gt, Ft, r0, rLTD, tsample = True, fs = 8):
    nt = Ft.size
    x_fpt = 1.05
    fig, ax = plt.subplots(dpi = 120, figsize = (3,3))
    
    points = np.array([Ft, Gt]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0,100)
    lc = LineCollection(segments, cmap='viridis', norm=norm, joinstyle='round', capstyle = 'round')
    # Set the values used for colormapping
    lc.set_array(np.linspace(0,100,nt+1)[:-1])
    lc.set_linewidth(3)
    lc.set_alpha(0.5)
    trace = ax.add_collection(lc)
    fig.colorbar(trace, ax=ax, label = 'time (%)', shrink = 0.5, anchor = (0,0))   
    if tsample:
        ax.plot(Ft, Gt, 'sr', ms = 0.5, alpha = 0.5, label = 'trace sample')
    ymax = Gt.max()*1.2
    r0_rLTD = r0/rLTD
    F = np.linspace(0,Ft.max()*1.2,200*int(r0_rLTD))
    line0 = ax.plot(F, np.zeros_like(F), '--', lw = 1, label = 'non-isolated,\n$r^\\ast=0$', color = 'gray')
    line1 = ax.plot(F, -4/27*r0_rLTD*r0_rLTD*F, ':', lw = 1, label = 'degenerate,\n$r^\\ast=2/3r_0$', color = 'gray')
    degen_y = lambda f: -4/27*r0_rLTD*r0_rLTD*f
    ax.text(F.max()*x_fpt, ymax*0.06, 'single\n fp', fontsize = fs-1, fontstyle = 'italic', horizontalalignment='right')
    ax.text(F.max()*x_fpt, degen_y(F.max())/3*2.4, 'double\n fp', fontsize = fs-1, fontstyle = 'italic', horizontalalignment='right')
    ax.text(F.max()*x_fpt, degen_y(F.max())*1.3, 'zero fp', fontsize = fs-1, fontstyle = 'italic', horizontalalignment='right')

    G_tau0 = np.array([get_Gtau(0, f, r0_rLTD) for f in F])
    G_tau1 = np.array([get_Gtau(1, f, r0_rLTD) for f in F])
    if not np.isnan(G_tau1).all():
        F1 = F[np.logical_not(np.isnan(G_tau1))].min()
        ax.text(F1 + F.max()*0.1, ymax*0.2, 'Unstable', fontsize = fs, horizontalalignment='left')
        ax.text(F1 - F.max()*0.1, ymax*0.2, 'Stable', fontsize = fs, horizontalalignment='right')
        line4 = ax.plot(F, G_tau1,  '-.k', lw = 1)
    else:
        ax.text(F.max()*0.3, ymax*0.3, 'Stable', fontsize = fs+3, horizontalalignment='left')
    if not np.isnan(G_tau0).all():
        F0 = F[np.logical_not(np.isnan(G_tau0))].min()
        ax.text(F0, degen_y(F0*0.6)/3*2.5, '+Saddle', fontsize = fs-1, horizontalalignment='center', verticalalignment = 'center', alpha = 1.0, color = 'gray')
        line3 = ax.plot(F, G_tau0,  '-.k', lw = 1)
    else:
        ax.text(F.max()*0.45, degen_y(F.max()*0.5)/3*2, '+Saddle', fontsize = fs+0.5, horizontalalignment='left', alpha = 1.0, color = 'gray')
    
    ymin = degen_y(F.max())*1.5
    ax.set_ylim(top=ymax/1.2*1.05, bottom = ymin)
    #ax.plot([0,0], [ymin, ymax], '-k', lw = 1, alpha = 0.5, label = 'Line attr.,\n$r^\\ast=0$')
    #ax.plot([0,0], [ymin, ymax], '-k', lw = 1, alpha = 0.5, label = 'Line attr.')
    ax.set_xlim(left=-Ft.max()*0.01)
    ax.set_xlabel('input gain G(t)', fontsize = fs+3)
    ax.set_ylabel('flux F(t)', fontsize = fs+3)
    #ax.legend(loc = 'best', bbox_to_anchor=(0.45, 0.85), fontsize = 'xx-small')
    #ax.legend(loc = 'outside upper right', fontsize = fs-1)
    #ax.legend(loc = 'best', fontsize = 10)
    ax.spines[['right', 'top']].set_visible(False)
    ax.legend(loc = 'upper left', bbox_to_anchor=(0.75, 1), fontsize = fs-1)
    return fig, ax


# In[7]:


r0 = 6
rLTD = 0.3
gL = 40 
nt = 256
l = 8
d = 4
v = 3.2
A = 0.002
rx = 20
k = 0.114*0.017*A
L = 2*l*np.sqrt(2) + 2*d
T = L/v
Ft = np.zeros(nt)
Gt = np.zeros(nt)
for i in range(nt):
    Ft[i] = k*rx*rx*int_form(T*i/(nt-1), v=v, d=d, l=l)/gL
    x_bot, x_top = get_xrange(T*i/(nt-1),v,d,l)
    Gt[i] = 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL
p = np.linspace(0.05,0.95,n)
F_tm = k*rx*rx*int_form(T*p, v=v, d=d, l=l)/gL
x_bot, x_top = get_xrange(T*p, v, d, l)
G_tm = 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL

fig, ax = stability_diagram(Gt, Ft, r0, rLTD, fs = 10)
ax.plot(F_tm, G_tm, '*k', ms = 5, alpha = 0.8)
fig.savefig(f'F-G_phase_diagram-l{l}-d{d}_match.png')
fig.savefig(f'F-G_phase_diagram-l{l}-d{d}_match.svg')


# In[8]:


def r_tau(r, r0, f):
    return 2*f*r - r*r/r0*f - 1
        
def temporal_trace(r0, Ft, Gt, r, rbar, t, fs = 10, ax = None, ax2 = None):
    if ax is None and ax2 is None:
        raise Exception('no axes provided')
    r2 = np.array([solve_for_fp(r0, Ft[i], Gt[i], pInfo = False, ret_nan = True)[0][-1] if -Gt[i] < 4/27*r0*r0*Ft[i] else np.nan for i in range(len(t))])
    nnan_pick = np.logical_not(np.isnan(r2))
    if np.sum(nnan_pick) > 0:
        r2_max =  r2[nnan_pick].max()
    else:
        r2_max = 0
    stable_pick = nnan_pick.copy()
    stable_pick[nnan_pick] = r_tau(r2[nnan_pick], r0, Ft[nnan_pick]) <= 0
    unstable_pick = nnan_pick.copy()
    unstable_pick[nnan_pick] = r_tau(r2[nnan_pick], r0, Ft[nnan_pick]) > 0
    r3 = r2.copy()
    r3[stable_pick] = np.nan #unstable
    r2[unstable_pick] = np.nan #stable
    
    r1 = np.array([solve_for_fp(r0, Ft[i], Gt[i], pInfo = False, ret_nan = True)[0][0] if -Gt[i] < 4/27*r0*r0*Ft[i] and Gt[i] <= 0 else np.nan for i in range(len(t))])
    if np.sum(np.logical_not(np.isnan(r1))) > 0:
        r1_max =  r1[np.logical_not(np.isnan(r1))].max()
    else:
        r1_max = 0
        
    if ax is not None:
        ax.plot(t,r, '-r', label = 'r')
        ax.plot(t,rbar, '-b', label = '$\\bar{r}$')
        ax.plot(t,r1, ':k', label = 'saddle')
        if not np.isnan(r2).all():
            ax.plot(t,r2, '--k', label = 'stable spiral')
        if not np.isnan(r3).all():
            ax.plot(t,r3, '-.', color = 'gray', label = 'unstable spiral')
        ax.set_ylabel('Rate (Hz)', fontsize = fs)
        ax.legend(fontsize = fs-2)
        ymax = r.max()*1.2
        r1[r1>ymax] = np.nan
        r2[r2>ymax] = np.nan
        r3[r3>ymax] = np.nan
        ax.set_ylim(0, ymax)
        ax.set_xlabel('t (s)', fontsize = fs)

    if ax2 is not None:
        nt = t.size
        points = np.array([t, r, rbar]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap=plt.get_cmap('viridis')
        colors=[cmap(float(ii)/(nt-1)) for ii in range(nt-1)]
        for ii in range(nt-2,-1,-1):
            if not np.isnan(r1[ii:ii+2]).all():
                ax2.plot(t[ii:ii+2],r1[ii:ii+2],r1[ii:ii+2], ':k', lw = 1, alpha = 1.0)
            if not np.isnan(r2[ii:ii+2]).all():             
                ax2.plot(t[ii:ii+2],r2[ii:ii+2],r2[ii:ii+2], '--k', lw = 1, alpha = 1.0)
            if not np.isnan(r3[ii:ii+2]).all():             
                ax2.plot(t[ii:ii+2],r3[ii:ii+2],r3[ii:ii+2], '-.', color = [0.2]*3, lw = 1, alpha = 1.0)
            segii = segments[ii]
            lii, = ax2.plot(segii[:,0], segii[:,1], segii[:,2], color=colors[ii], linewidth=3, alpha = 0.5)
            lii.set_solid_capstyle('round')
            lii.set_solid_joinstyle('round')
        norm = plt.Normalize(0,100)
        lc = Line3DCollection(segments, cmap='viridis', norm=norm)
        trace = ax2.add_collection(lc)
        plt.colorbar(trace, ax=ax2, label = 'time (%)', shrink = 0.25)  
        trace.remove()
        
        ax2.set_ylim(0, ymax)
        ax2.set_zlim(0, ymax)
        ax2.set_xlabel('t (s)', fontsize = fs, labelpad = 6)
        ax2.set_ylabel('r (Hz)', fontsize = fs)
        ax2.set_zlabel('$\\bar{r}$ (Hz)', fontsize = fs)
        ax2.view_init(elev=12, azim=200) 
        ax2.set_box_aspect((10,4,4))


# In[9]:


nt = 9000
l = 24
d = 4
v = 3.2
r0 = 6
rLTD = 0.3
rx = 20
gL = 40
w0 = 0.1
A = 0.01
k = 0.114*0.017
nT = 1
tau = 1
G_perc = 1.0
wt, r, rbar, dt, F, F0, G = iter_sweep('exact2', nT = nT, w0 = w0, l = l, d = d, v = v, k = k, A = A, rLTD = rLTD, r0 = r0, rx = rx, tau = tau, tau_m = tau_m, gL = gL, cap = cap, nx = nx, nt = nt, plot = True, theme = 'check', fork_ivp = fork_ivp, shape = 'circle', reverse = 1, norm_w = 0, figsize = (10, 6), return_FG = True, G_perc = G_perc)
nt = r.size
t = np.arange(nt)*dt
print(r.max(), rbar.max())
dwdt = (r*(r-rbar*rbar/r0))
fig = plt.gcf()
ax = fig.get_axes()


# In[10]:


fs = 12
print(ax)
for _ax in ax:
    _ax.set_xlabel(_ax.get_xlabel(), fontsize = fs-1)
    _ax.tick_params(axis = 'both', labelsize = fs-2)
    
ax[1].set_title(rf'single sweep: $\rightarrow$, $w_0=0.1$, $d={d}$, $l={l}$', fontsize = fs)
ax[1].legend(fontsize = fs-2)
ax[1].tick_params(labelbottom = False)
ax[2].tick_params(labelbottom = False)
ax[1].set_ylabel(ax[1].get_ylabel(), fontsize = fs-1, labelpad = 10.0)
ax[2].set_ylabel(ax[2].get_ylabel(), fontsize = fs-1)
ax[3].set_ylabel('G(t) dw/dt', fontsize = fs-1, labelpad = 2)
ax[4].set_ylabel('F(t)', fontsize = fs-1, labelpad = 5)
#box = ax[3].get_position()
#ax[3].set_position([box.x0, box.y0+0.06, box.width, box.height])
#box = ax[0].get_position()
#ax[0].set_position([box.x0, box.y0+0.06, box.width, box.height])
ax[0].set_title('')
ax[0].set_ylabel('weight', fontsize = fs-1, labelpad = 8)
ax[0].plot([0,2*l], [w0, w0], ':', color = 'grey')
fig.set_size_inches((5,5))
fig.savefig('1d_periodic.png')
fig.savefig('1d_periodic.svg')
fig


# In[11]:


fig = plt.figure(dpi = 200, figsize = (10,6)) 
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection = '3d')
fig, axs = temporal_trace(r0/rLTD, F0, G, r, rbar, t, fs = 10, ax = ax, ax2 = ax2)
box = ax.get_position()
ax.set_position([box.x0 + 0.15, box.y0+0.2, box.width*0.7, box.height*0.4])
fig.savefig('3d_trace_phase_diagram.png')
fig.savefig('3d_trace_phase_diagram.svg')


# In[12]:


fig, ax = stability_diagram(G, F0, r0, rLTD, tsample = False)
fig.savefig(f'F-G_phase_diagram-l{l}-d{d}.png')
fig.savefig(f'F-G_phase_diagram-l{l}-d{d}.svg')


# In[13]:


v = 3.2 * 20/16
w0 = 0.1
d = 4
l = 8
L = 2*l*np.sqrt(2) + 2*d
#l0 = l*(np.sqrt(2)-1)
nt = 1001
fig = plt.figure(figsize = (9, 4), dpi = 200)

label_FS = 12
label_fs = 6
ax = fig.add_subplot(234)
###
v0 = 0.5*v
v1 = v
v2 = 2*v
T = L/v0
t0 = np.linspace(0,T,nt)
T = L/v1
t1 = np.linspace(0,T,nt)
T = L/v2
t2 = np.linspace(0,T,nt)
x_bot, x_top = get_xrange(t0,v0,d,l)
ax.plot(t0*v0-L/2, 2*v0*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, ':k', alpha = 0.8, label = r'$v = 0.5\times v^*$')
x_bot, x_top = get_xrange(t1,v1,d,l)
ax.plot(t1*v1-L/2, 2*v1*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, '--k', alpha = 0.8, label = r'$v = 1\times v^*$')
x_bot, x_top = get_xrange(t2,v2,d,l)
ax.plot(t2*v2-L/2, 2*v2*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, '-k', alpha = 0.8, label = r'$v = 2\times v^*$')
#ax.plot(np.array([0, L]) - 2*l*(np.sqrt(2)-1)/2, [0, 0], '-', color = 'gray', lw = 1, alpha = 0.5)
ax.legend(fontsize = label_fs+2)

ax.set_ylabel('flux F(t)', fontsize = label_FS)
#ax.set_xlabel('wave front (#LGN)', fontsize = label_FS)
ax.set_xlabel(r'rel. wave front, $x$ (#LGN)', fontsize = label_FS)

ax = fig.add_subplot(232)
ax2 = fig.add_subplot(235)
d0 = 2
L = 2*l*np.sqrt(2) + 2*d0
T = L/v
t0 = np.linspace(0,T,nt)
ax.plot(t0*v-L/2, k*rx*rx*int_form(t0, v=v, d=d0, l=l)/gL, ':k', alpha = 0.8, label = r'$d=d^* - 2$')
x_bot, x_top = get_xrange(t0,v,d0,l)
ax2.plot(t0*v-L/2, 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, ':k', alpha = 0.8, label = r'$d=d^* - 2$')

d1 = 4
L = 2*l*np.sqrt(2) + 2*d1
T = L/v
t1 = np.linspace(0,T,nt)
ax.plot(t1*v-L/2, k*rx*rx*int_form(t1, v=v, d=d1, l=l)/gL, '--k', alpha = 0.8, label = r'$d = d^*$')
x_bot, x_top = get_xrange(t1,v,d1,l)
ax2.plot(t1*v-L/2, 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, '--k', alpha = 0.8, label = r'$d = d^*$')

d2 = 6
L = 2*l*np.sqrt(2) + 2*d2
T = L/v
t2 = np.linspace(0,T,nt)
ax.plot(t2*v-L/2, k*rx*rx*int_form(t2, v=v, d=d2, l=l)/gL, '-k', alpha = 0.8, label = r'$d = d^* + 2$')
x_bot, x_top = get_xrange(t2,v,d2,l)
ax2.plot(t2*v-L/2, 2*v*rx*w0*(half_form(x_top, l) - half_form(x_bot, l))/gL, '-k', alpha = 0.8, label = r'$d = d^* + 2$')

ax.legend(fontsize = label_fs+2)
ax2.legend(fontsize = label_fs+2)
ax.set_ylabel('input gain G(t)', fontsize = label_FS)
ax2.set_ylabel('flux F(t)', fontsize = label_FS)
#ax.set_xlabel('wave front (#LGN)', fontsize = label_FS)
ax.set_xlabel('rel. wave front, $x$ (#LGN)', fontsize = label_FS)
ax2.set_xlabel('rel. wave front, $x$ (#LGN)', fontsize = label_FS)

ax = fig.add_subplot(233)
ax2 = fig.add_subplot(236)
nx = 101
xbin = np.linspace(0,2*l,nx)
L = 2*l*np.sqrt(2) + 2*d
T = L/v
t = np.linspace(0,T,nt)
x_bot, x_top = get_xrange(t,v,d,l)
x = (xbin[1:] + xbin[:-1])/2
w_dist0 = lambda x: w0*np.ones(x.shape)
print(np.sum(w_dist0(x)))
cut_norm_dist = lambda x, sigma: w0*2*l/np.sqrt(2*np.pi)/sigma*np.exp(-0.5*np.power((x-l)/sigma,2))/sp.special.erf(l/np.sqrt(2)/sigma)
ax.plot(x, w_dist0(x), ':k', alpha = 0.8, label = r'std.$ = \infty$')
ax2.plot(t*v-L/2, 2*v*rx*(w_dist0(x_top)*half_form(x_top, l) - w_dist0(x_bot)*half_form(x_bot, l))/gL, ':k', alpha = 0.8, label = r'std.$ = \infty$')
sig = 5
w_dist1 = lambda x: cut_norm_dist(x, sig)
print(np.sum(w_dist1(x)))
ax.plot(x, w_dist1(x), '--k', alpha = 0.8, label = r'std.$ = 5$')
ax2.plot(t*v-L/2, 2*v*rx*(w_dist1(x_top)*half_form(x_top, l) - w_dist1(x_bot)*half_form(x_bot, l))/gL, '--k', alpha = 0.8, label = r'std.$ = 5$')
sig = 3
w_dist2 = lambda x: cut_norm_dist(x, sig)
print(np.sum(w_dist2(x)))
ax.plot(x, w_dist2(x), '-k', alpha = 0.8, label = r'std.$ = 1$')
ax2.plot(t*v-L/2, 2*v*rx*(w_dist2(x_top)*half_form(x_top, l) - w_dist2(x_bot)*half_form(x_bot, l))/gL, '-k', alpha = 0.8, label = r'std.$ = 1$')
ax.legend(fontsize = label_fs+2, loc = 'upper right')
ax2.legend(fontsize = label_fs+2)
ax.set_xlabel(r'$x$ (#LGN)', fontsize = label_FS)
ax.set_ylabel('conn. weight', fontsize = label_FS)
ax2.set_xlabel(r'rel. wave front, $x$ (#LGN)', fontsize = label_FS)
ax2.set_ylabel('flux F(t)', fontsize = label_FS)

fig.tight_layout()
fig.savefig('mechanism_illustrate.png')
fig.savefig('mechanism_illustrate.svg')


# In[ ]:




