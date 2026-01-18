import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.gridspec as gs
import matplotlib as mpl
from scipy.optimize import brentq, newton 
#import mpl_toolkits.mplot3d.art3d.Line3DCollection
import sys
from reduced_model_utils import *
mpl.rcParams.update({'font.family': 'CMU Sans Serif', 'axes.unicode_minus' : False})
mpl.rcParams.update({'mathtext.fontset': 'cm', 'mathtext.default':'regular'})

def phase_plane(dF, x, y, ax = None, lw = 0.8, d = 0.75, color = 'gray'):
    X, Y = np.meshgrid(x,y)
    U, V = dF(X, Y)
    if ax is None:
        fig, ax = plt.subplots()
        ax_is_None = True
    else:
        ax_is_None = False
    ax.streamplot(X, Y, U, V, linewidth = lw, density = d, arrowsize = 0.5, zorder = 0, color = color)
    if ax_is_None:
        return ax
    
def nullcline2(r, w, uniform_w, half_fxt, iint_fxt, t, k, l, d, v, r0, rx, gL):
    bot, top = get_xrange(t,v,d,l)
    Gx = iint_fxt(top,l) - iint_fxt(bot,l)
    if Gx == 0:
        if hasattr(r, '__len__'):
            rbar = np.zeros_like(r)
            rbar[:] = np.nan
            return rbar, np.nan
        else:
            return np.nan, np.nan
        
    if uniform_w:
        if hasattr(w, '__len__'):
            fxw = 2*(half_fxt(top, l) - half_fxt(bot, l))*w[0]
        else:
            fxw = 2*(half_fxt(top, l) - half_fxt(bot, l))*w
    else:
        nx = len(w)
        x = np.linspace(0,2*l,nx)
        w_top = np.interp(top, x, w)
        w_bot = np.interp(bot, x, w)
        fxw = 2*(half_fxt(top, l)*w_top - half_fxt(bot, l)*w_bot)
    fxw /= gL
    Gx /= gL
    if hasattr(r, '__len__'):
        rbar = np.zeros_like(r)
        _tmp = np.zeros_like(r)
        rpick = r!=0
        _tmp[rpick] = v*fxw/(k*rx*Gx*r[rpick]) + r[rpick]
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
            ro = np.sqrt(-v*fxw/(k*rx*Gx))
        return rbar, ro
    else:
        if r == 0:
            return np.nan
        else:
            _tmp = v*fxw/(k*rx*Gx*r) + r
            if _tmp >= 0:
                return np.sqrt(_tmp*r0)
            else:
                return np.nan

def get_dF2(w, uniform_w, half_fxt, iint_fxt, t, k = 0.114*0.017*0.025, r0 = 6, l = 8, d = 3, v = 3.2, rx = 20, gL = 40, tau = 1):
    def dF(r, r_bar):
        Gt, bot, top = deltaW_Area(r, r_bar, iint_fxt, t, k, l, d, v, r0, rx, False)
        Ft = deltaArea_W(half_fxt, w, uniform_w, rx, v, l, bot = bot, top = top)
        return (Ft + Gt)/gL, r - r_bar
    return dF

def get_Ftau(which, g, r0_rLTD):
    if r0_rLTD*g >= 1:
        if which == 1:
            tau_r = r0_rLTD + np.sqrt(g*g*r0_rLTD*r0_rLTD - g*r0_rLTD)/g
        else:
            tau_r = r0_rLTD - np.sqrt(g*g*r0_rLTD*r0_rLTD - g*r0_rLTD)/g
        if tau_r*g*(3*tau_r/r0_rLTD-2) <= 0:
            return np.nan
        else:
            return np.power(tau_r,2)*(tau_r - r0_rLTD)/r0_rLTD*g
    else:
        return np.nan

def stability_diagram(Gt, Ft, r0, rLTD, tsample = 20, fs = 8, ax = None, pInfo = True):
    nt = Gt.size
    x_fpt = 0.9
    if ax is None:
        ret_fig = True
        fig, ax = plt.subplots(dpi = 120, figsize = (3,3))
    else:
        ret_fig = False
    
    nseg = 250
    points = np.array([np.interp(np.linspace(0,1,nseg+1), np.linspace(0,1,nt), Gt), np.interp(np.linspace(0,1,nseg+1), np.linspace(0,1,nt), Ft)]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0,100)
    lc = LineCollection(segments, cmap='viridis', norm=norm, joinstyle='round', capstyle = 'round')
    # Set the values used for colormapping
    lc.set_array(np.linspace(0,100,nseg+1)[:-1])
    lc.set_linewidth(3)
    lc.set_alpha(0.6)
    trace = ax.add_collection(lc)
    if pInfo:
        plt.colorbar(trace, ax=ax, label = 'time (%)', shrink = 0.5, anchor = (0,0))   
    if tsample is not None:
        if not hasattr(tsample, '__len__'):
            fpick = np.linspace(0,1,tsample+1)
        else:
            fpick = tsample
        ax.plot(np.interp(fpick, np.linspace(0,1,nt), Gt), np.interp(fpick, np.linspace(0,1,nt), Ft), '.', c='r', ms = 8, alpha = 0.8, label = 'trace sample')
    ymax = Ft.max()*1.2
    r0_rLTD = r0/rLTD
    G = np.linspace(0, Gt.max()*1.2, 200*int(r0_rLTD))
    line0 = ax.plot(G, np.zeros_like(G), '--', lw = 1, label = 'non-isolated,\n$r^\\ast=0$', color = 'gray')
    line1 = ax.plot(G, -4/27*r0_rLTD*r0_rLTD*G, ':', lw = 1, label = 'degenerate,\n$r^\\ast=2/3r_0$', color = 'gray')
    degen_y = lambda g: -4/27*r0_rLTD*r0_rLTD*g
    #ax.text(G.max()*x_fpt, ymax*0.1, 'single\n FP.', fontsize = fs-1, fontstyle = 'italic', horizontalalignment='right')
    ax.text(G.max()*x_fpt, ymax*0.1, '1 FP.', fontsize = fs-2, fontstyle = 'italic', horizontalalignment='left')
    #ax.text(G.max()*x_fpt, degen_y(G.max())/3*2.5, 'double\n FP.', fontsize = fs-1, fontstyle = 'italic', horizontalalignment='right')
    ax.text(G.max()*x_fpt, degen_y(G.max())/3*2, '2 FP.\'s', fontsize = fs-2, fontstyle = 'italic', horizontalalignment='left')
    #ax.text(G.max()*x_fpt, degen_y(G.max())*1.5, 'zero FP.', fontsize = fs-1, fontstyle = 'italic', horizontalalignment='right')
    ax.text(G.max()*x_fpt, degen_y(G.max())*1.5, '0 FP.', fontsize = fs-2, fontstyle = 'italic', horizontalalignment='left')

    G_tau0 = np.array([get_Ftau(0, g, r0_rLTD) for g in G])
    G_tau1 = np.array([get_Ftau(1, g, r0_rLTD) for g in G])
    if not np.isnan(G_tau1).all():
        G1 = G[np.logical_not(np.isnan(G_tau1))].min()
        ax.text(G1 + G.max()*0.1, ymax*0.2, 'Unstable', fontsize = fs, horizontalalignment='left')
        ax.text(G1 - G.max()*0.07, ymax*0.2, 'Stable', fontsize = fs, horizontalalignment='right')
        line4 = ax.plot(G, G_tau1,  '-.k', lw = 1)
    else:
        ax.text(G.max()*0.3, ymax*0.3, 'Stable', fontsize = fs+3, horizontalalignment='left')
    if not np.isnan(G_tau0).all():
        G0 = G[np.logical_not(np.isnan(G_tau0))].min()
        #ax.text(G0, degen_y(G0*0.6)/3*2, '+Saddle', fontsize = fs, horizontalalignment='center', verticalalignment = 'center', alpha = 1.0, color = 'gray')
        ax.text(G0*1.15, degen_y(G0*0.6)/3*2.8, '+Saddle', fontsize = fs, horizontalalignment='left', verticalalignment = 'center', alpha = 1.0, color = 'gray')
        line3 = ax.plot(G, G_tau0,  '-.k', lw = 1)
    else:
        ax.text(G.max()*0.4, degen_y(G.max()*0.5)/3*2, '+ Saddle', fontsize = fs+1, horizontalalignment='left', alpha = 1.0, color = 'gray')

    ymin = min(degen_y(G.max())*2, -ymax*0.2)
    ax.set_ylim(top=ymax/1.2*1.05, bottom = ymin)
    ax.set_xlim(left=-Gt.max()*0.01, right = Gt.max()*1.3)
    ax.set_xlabel('input gain, G(t)', fontsize = fs)
    ax.set_ylabel('flux, F(t)', fontsize = fs)
    ax.spines[['right', 'top']].set_visible(False)
    if pInfo:
        ax.legend(loc = 'upper left', bbox_to_anchor=(0.75, 1), fontsize = fs-1, handlelength=1)

    if ret_fig:
        return fig, ax

def solve_for_fp(r0, Gt, Ft, tol = 1e-10, pInfo = True, ret_nan = False, pw_tol = 1e-14):
    if Gt == 0.0:
        if pInfo:
            print('no fp')
        if Ft == 0:
            if ret_nan:
                return [np.nan], ['Line']
            else:
                return None, ['Line']
        else:
            if ret_nan:
                return [np.nan], ['None']
            else:
                return None, ['None']
    elif Ft == 0:
        if pInfo:
            print('zero flux')
        if r0 * Gt > 1:
            return [0.0, r0], ['Non-isolated',  'Unstable']
        else:
            return [0.0, r0], ['Non-isolated',  'Stable']
    else:
        f = lambda r: np.power(r,3) - r*r*r0 - Ft/Gt*r0
        fprime = lambda r: 3*r*r - 2*r*r0
        fprime2 = lambda r: 6*r - 2*r0
        tau = lambda r: 2*Gt*r - r*r/r0*Gt - 1
        if pw_tol > 0 and np.abs(Ft + 4/27*r0*r0*Gt) < pw_tol: # transcritical bifurcation
            if pInfo:
                print('transcritical')
            return [[2/3*r0], 'Degenerated']
        elif Ft + 4/27*r0*r0*Gt < 0:
            if pInfo:
                print('no fp')
            if ret_nan:
                return [np.nan], ['None']
            else:
                return None, ['None']
        elif Ft > 0: # single fp
            if pInfo:
                print(f'f(0) = {f(0)}, f(2/3*r0):= f({2/3*r0})= {f(2/3*r0)}')
            root = newton(f, 2/3*r0 + tol, fprime, fprime2 = fprime2, rtol = tol, maxiter = 200)
            if pw_tol > 0 and np.abs(tau(root)) < pw_tol:
                if pInfo:
                    print(f'center single fp = {[root, f(root)]}')
                return [root], ['Center']
            elif tau(root) < 0:
                if pInfo:
                    print(f'Stable single fp = {[root, f(root)]}')
                return [root], ['Stable']
            else:
                if pInfo:
                    print(f'Unstable single fp = {[root, f(root)]}')
                return [root], ['Unstable']
        else: # two fp
            if pInfo:
                print(f'brentq brackets: f(0) = {f(0)}, f(2/3*r0) = {f(2/3*r0)}, f(r0) = {f(r0)}')
            root1 = brentq(f, 0, 2/3*r0)
            root2 = brentq(f, 2/3*r0, 4/3*r0)
            if pInfo:
                print(f'double fp = {[root1, f(root1)]}; {[root2, f(root2)]}')
                print(f'tau = {tau(root2)}')
            if pw_tol > 0 and np.abs(tau(root2)) < pw_tol:
                return [root1, root2], ['Saddle', 'Center']
            if tau(root2) < 0:
                return [root1, root2], ['Saddle', 'Stable']
            else:
                return [root1, root2], ['Saddle', 'Unstable']

def get_fp_type(r0, g, f):
    fp, _type = solve_for_fp(r0, g, f, pInfo = False, ret_nan = True, pw_tol = 0) 
    if _type[0] == 'None' or _type[0] == 'Line':
        fp_type = 'None'
        true_type = ['None']
    else:
        assert(np.logical_not(np.isnan(fp).any()))
        if len(_type) == 1:
            fp_type = _type[0]
        else:
            fp_type = f'{_type[0]}+{_type[1]}'
            true_type = _type
        true_type = _type
    return fp_type, true_type, fp

def fp_t_tuple(t, pit, cit, w, uniform_w, prev_type, half_fxt, iint_fxt, r0, rx, v, l, k, d, gL, fp, fi, nt = 50):
    _t = np.linspace(t[pit], t[cit], nt)
    if uniform_w or len(w.shape) == 1:
        _w = w
    else:
        _w = w[:,pit:cit+1]

    bot, top = get_xrange(_t[1:],v,d,l)
    g = k*rx*rx*(iint_fxt(top,l) - iint_fxt(bot,l))/gL
    f = deltaArea_W(half_fxt, _w, uniform_w, rx, v, l, bot = bot, top = top, skip_zero = True)/gL
    for i in range(1,nt):
        current_type, types, fps = get_fp_type(r0, g[i-1], f[i-1])
        if current_type != prev_type:
            if current_type != 'None':
                for _type, _fp in zip(types, fps):
                    if _type not in fp.keys():
                        fp[_type] = [([_t[i]], [_fp])]
                        fi[_type] = 0
                    else:
                        fp[_type].append(([_t[i]], [_fp]))
                        fi[_type] += 1
            prev_type = current_type
        else:
            if current_type != 'None':
                for _type, _fp in zip(types, fps):
                    fp[_type][fi[_type]][0].append(_t[i])
                    fp[_type][fi[_type]][1].append(_fp)


def solve_fp_t(r0, Gt, Ft, w, uniform_w, t, half_fxt, iint_fxt, rx, v, l, k, d, gL):
    fp = {} # {'type': [([t],[fp]), ...]}
    fi = {} # index of fp
    prev_type, types, fps = get_fp_type(r0, Gt[0], Ft[0])
    if prev_type != 'None':
        for _type, _fp in zip(types, fps):
            fp[_type] = [([t[0]], [_fp])]
            fi[_type] = 0

    for i in range(1,len(t)):
        current_type, types, fps = get_fp_type(r0, Gt[i], Ft[i])
        if current_type != prev_type:
            # fill-in gaps if possible
            #fp, fi = fp_t_tuple(t, i-1, i, w, uniform_w, prev_type, half_fxt, iint_fxt, r0, rx, v, l, k, d, gL, fp, fi)
            fp_t_tuple(t, i-1, i, w, uniform_w, prev_type, half_fxt, iint_fxt, r0, rx, v, l, k, d, gL, fp, fi)
            prev_type = current_type
        else:
            if current_type != 'None':
                for _type, _fp in zip(types, fps):
                    fp[_type][fi[_type]][0].append(t[i])
                    fp[_type][fi[_type]][1].append(_fp)

    return fp

def phase_portrait(ax, p, rx, r0, rLTD, k, v, d, l, gL, tau, half_form, iint_form, int_form, T, w, r, rbar, t, plot_xlabel = True, plot_ylabel = True, scale = 2, G = None, Ft = None, label_fs = 9, label_FS = 10):
    if scale == 0:
        scale = 1
    elif scale > 0:
        rmax = scale*r.max()
    else:
        rmax = -scale

    R = np.linspace(0, rmax, 100)
    RBAR = np.linspace(0, rmax, 100)
    if G is None:
        G = k*rx*rx*int_form(T*p, v=v, d=d, l=l)/gL
    else:
        print(f'G = {G}, G\' = {k*rx*rx*int_form(T*p, v=v, d=d, l=l)/gL}')

    mks = {'Stable': '.k', 'Unstable': '*k', 'Saddle':'dm', 'Degenerated': 'sm', 'Non-isolated':'^m'}

    null_color1 = np.array([225,139,107], dtype = float)/255
    null_color2 = np.array([255,165,0], dtype = float)/255
    stream_color = [0.4, 0.4, 0.4]
    if hasattr(w, 'shape'):
        uniform_w = False
        if len(w.shape) == 2: 
            wt = np.zeros(w.shape[0])
            for i in range(w.shape[0]):
                wt[i] = np.interp(T*p, t, w[i,:])
        else:
            wt = w.copy()
    else:
        wt = w
        uniform_w = True

    phase_plane(get_dF2(wt, uniform_w, half_form, iint_form, T*p, k = k, r0 = r0/rLTD, l = l, d = d, v = v, rx = rx, tau = tau, gL = gL), R, RBAR, ax = ax, color = stream_color)
    _r = np.linspace(0, R[-1], 1000)
    _rbar, ro = nullcline2(_r, wt, uniform_w, half_form, iint_form, T*p, k, l, d, v, r0/rLTD, rx, gL)
    ax.plot(_r, _rbar, c = null_color1, lw = 2, alpha = 1, zorder = 0)
    x_bot, x_top = get_xrange(T*p,v,d,l)
    if Ft is None:
        Ft = deltaArea_W(half_form, wt, uniform_w, rx, v, l, bot = x_bot, top = x_top)/gL
    else:
        print(f'Ft = {Ft}, Ft\' = {deltaArea_W(half_form, wt, uniform_w, rx, v, l, bot = x_bot, top = x_top)/gL}')
    if Ft <= 1e-14:
        if np.isnan(ro):
            ro = 0
        if ro < r.max():
            iro = np.nonzero(ro - _r <= 0)[0][0]
            __r = np.insert(_r, max([0,iro]), ro)
            _rbar[iro:][np.isnan(_rbar[iro:])] = 0
            __rbar = np.insert(_rbar, max([0,iro]), 0)
            #ax.plot(__r, __rbar, 'g', lw = 2, alpha = 1) 
            ax.plot(__r, __rbar, c = null_color1, lw = 2, alpha = 1) 
        #ax.plot(ro, 0, 'og', ms = 8, fillstyle = 'none', mew = 2, lw = 2, alpha = 1, zorder = 0) 
    ax.plot(_r, _r, c = null_color2, lw = 2, alpha = 1, zorder = 0) 
    roots, types = solve_for_fp(r0/rLTD, G, Ft) 
    if roots is not None: 
        for root, typ in zip(roots, types): 
            ax.plot(root, root, mks[typ], mfc = 'none', mew = 1.5) 
            #ax.set_title(f'{p*100:.0f}% T \n G(t) = {G:.2e}\n F(t)={Ft:.1f}', fontsize = 'xx-small') 
            ax.set_title(f'{p*100:.0f}% T', fontsize = label_FS + 2) 
    else: 
        #ax.set_title(f'{p*100:.0f}% T \n G(t) = {G:.2e}\n F(t)={Ft:.1f}, {types}', fontsize = 'xx-small') 
        #ax.set_title(f'{p*100:.0f}% T\n{types}', fontsize = 'xx-small') 
        ax.set_title(f'{p*100:.0f}% T', fontsize = label_FS + 2) 
    if not plot_ylabel: 
        ax.set_yticklabels([]) 

    # plot actual vector <r, rbar>
    rp = np.interp(T*p, t, r)
    rbarp = np.interp(T*p, t, rbar)
    dt = t[1] - t[0]
    if T*p+dt > t[-1]:
        dr = rp - np.interp(T*p-dt, t, r)
        drbar = rbarp - np.interp(T*p-dt, t, rbar)
    else:
        dr = np.interp(T*p+dt, t, r) - rp
        drbar = np.interp(T*p+dt, t, rbar) - rbarp
    scale = (R[-1] - R[0])*0.15
    d = np.sqrt(dr*dr + drbar*drbar)
    dr = dr/d*scale
    drbar = drbar/d*scale
    #ax.arrow(rp, rbarp, dr, drbar, color = 'k', width = 0.1, length_includes_head=True, zorder = 2)
    ax.arrow(rp-dr*0.5, rbarp-drbar*0.5, dr, drbar, color = 'k', lw = 0.5, width = 0.2, head_width = 1.0, head_length = 1.0, zorder = 2, overhang = 0.35, length_includes_head = True)

    ax.set_xlim(R[0], R[-1]) 
    ax.tick_params(axis = 'both', labelsize = label_fs) 
    ax.set_ylim(RBAR[0], RBAR[-1]) 
    ax.set_aspect('equal') 
    if plot_ylabel:
        ax.set_ylabel('avg. FR (Hz)') 
    if plot_xlabel:
        ax.set_xlabel('FR (Hz)')

def r_tau(r, r0, g):
    return 2*g*r - r*r/r0*g - 1
        
def temporal_trace(r0, Gt, Ft, r, rbar, t, w, uniform_w, half_fxt, iint_fxt, rx, v, l, k, d, gL, fs = 10, ax = None, ax2 = None, xf = None, pscale = 1.5, plot_ylabel = True, plot_xlabel = True, pLeg = True, less = 0):
    if ax is None and ax2 is None:
        raise Exception('no axes provided')

    fp = solve_fp_t(r0, Gt, Ft, w, uniform_w, t, half_fxt, iint_fxt, rx, v, l, k, d, gL)

    ls = {'Saddle': ':', 'Stable': '--', 'Unstable': '-.', 'Non-isolated': '^'}
    color = {'Saddle': 'm', 'Stable': 'gray', 'Unstable': 'gray', 'Non-isolated': 'gray'}
    fp_label = {'Saddle': 'saddle FP.', 'Stable': 'stable FP.', 'Unstable': 'unstable FP.', 'Non-isolated': 'non-isolated FP.'}
    if xf is None:
        xf = lambda x: x
    if ax is not None:
        leg0 = []
        leg0.append(ax.plot(xf(t),r, 'k', label = 'FR.'))
        if less == 2:
            leg0.append(ax.plot(xf(t),rbar, ':k', label = 'avg. FR.'))
        thres_r = np.power(rbar,2)/r0
        leg0.append(ax.plot(xf(t), thres_r, 'r', label = 'thres. FR.'))
        if pscale > 0:
            ymax = max([r.max(), thres_r.max(), rbar.max()])*pscale
        else:
            ymax = -pscale
        print(f' ymax = {ymax}')

        fpmax = -1
        t_max = -1
        leg = []
        if less != 2:
            for fp_type in fp.keys():
                if fp_type == 'Non-isolated':
                    continue
                first_time = True
                coil = -1
                for seq in fp[fp_type]:
                    r_fp = np.array(seq[1], dtype = float)
                    fp_t = np.array(seq[0], dtype = float)
                    if coil > 0:
                        arg_fpmax = np.argmax(r_fp)
                        if r_fp[arg_fpmax] > fpmax:
                            fpmax = r_fp[arg_fpmax]
                            t_max = fp_t[arg_fpmax]
                    if not np.isnan(r_fp).all():
                        fp_line = ax.plot(xf(fp_t), r_fp, ls[fp_type], color = color[fp_type], label = fp_label[fp_type])
                        if first_time:
                            leg.append(fp_line)
                            first_time = False
                    coil += 1

            #ymax = max(ymax, fpmax*1.2)

        if plot_ylabel:
            ax.set_ylabel('Rate (Hz)', fontsize = fs)
        if pLeg:
            if ax2 is not None:
                if len(leg) > 0:
                    leg1 = ax.legend([h[0] for h in leg], [h[0].get_label() for h in leg], loc = 'upper left', fontsize = fs-2, labelcolor = 'gray', frameon = False, handlelength = 1.8, handletextpad = 0.5, bbox_to_anchor = (0.6, 0.78))
                    ax.add_artist(leg1)
                ax.legend([h[0] for h in leg0], [h[0].get_label() for h in leg0], fontsize=fs-2, loc = 'lower left', handlelength = 1.8, handletextpad = 0.5, frameon = False, bbox_to_anchor = (0.6, 0.7))
            else:
                if len(leg) > 0:
                    leg1 = ax.legend([h[0] for h in leg], [h[0].get_label() for h in leg], loc = 'upper right', fontsize = fs-2, labelcolor = 'gray', frameon = False, handlelength = 0.8, handletextpad = 0.5, bbox_to_anchor = (1.02, 1.05))
                    ax.add_artist(leg1)
                ax.legend([h[0] for h in leg0], [h[0].get_label() for h in leg0], fontsize=fs-3, loc = 'upper left', handlelength = 0.8, handletextpad = 0.5, frameon = False, bbox_to_anchor = (0, 1.05))
        ax.set_ylim(0, ymax)
        #ax.set_xlabel('t (s)', fontsize = fs)

    if ax2 is not None:
        nt = t.size
        points = np.array([xf(t), r, rbar]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        cmap=plt.get_cmap('viridis')
        trace_color=[cmap(float(ii)/(nt-1)) for ii in range(nt-1)]
        for fp_type in fp.keys():
            if fp_type == 'Non-isolated':
                continue
            for seq in fp[fp_type]:
                r_fp = np.array(seq[1], dtype = float)
                fp_t = np.array(seq[0], dtype = float)
                seg_pick = r_fp < ymax
                if seg_pick.any():
                    ax2.plot(xf(fp_t[seg_pick]),r_fp[seg_pick],r_fp[seg_pick], ls = ls[fp_type], color = color[fp_type], lw = 2, alpha = 1.0, zorder = 1)

        for ii in range(nt-2,-1,-1):
            segii = segments[ii]
            if segii[0,2] >  segii[0,1]:
                zorder = 2
            else:
                zorder = 0
            lii, = ax2.plot(segii[:,0], segii[:,1], segii[:,2], color=trace_color[ii], linewidth=3, alpha = 0.2, zorder = zorder)
            lii.set_solid_capstyle('round')
            lii.set_solid_joinstyle('round')
        norm = plt.Normalize(0,100)
        lc = Line3DCollection(segments, cmap='viridis', norm=norm)
        trace = ax2.add_collection(lc)
        cb = plt.colorbar(trace, ax=ax2, label = 'time (%)', shrink = 0.24)  
        cb.ax.tick_params(labelsize = fs-1)
        box = cb.ax.get_position()
        cb.ax.set_position([box.x0-0.05, box.y0 + box.height*0.3, box.width, box.height])
        trace.remove()
        ax2.set_ylim(0, ymax)
        ax2.set_zlim(0, ymax)
        #ax2.set_xlabel('t (s)', fontsize = fs, labelpad = 6)
        ax2.set_xlabel('wave front (#LGN)', fontsize = fs, labelpad = 10)
        ax2.set_ylabel('FR. (Hz)', fontsize = fs)
        ax2.zaxis.set_rotate_label(False)  
        ax2.set_zlabel('avg. FR. (Hz)', fontsize = fs, rotation = 90, labelpad = 1)
        ax2.tick_params(axis = 'x', pad = 0.3)
        ax2.tick_params(axis = 'y', pad = -0.1)
        ax2.tick_params(axis = 'z', pad = 0.3)
        ax2.view_init(elev = 12, azim = 200)
        ax2.set_box_aspect((10, 4, 4))
    if fpmax > -1:
        print(f'max fixed point in coil: {xf(t_max), fpmax}')
    return fp
