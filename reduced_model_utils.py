import numpy as np
from scipy.interpolate import interpn

def get_xrange(t,v,d,l, nd = 2, center_range = None, ret_vec = False):
    L = 2*l*np.sqrt(2) + 2*d
    if center_range is None:
        center_range = 2*d+2*l
    dL = (L - center_range)/2
    x = np.mod(v*t, L) - dL
    if hasattr(x, '__len__'):
        bot = x-nd*d
        bot[bot<0] = 0
        bot[bot>2*l] = 2*l
        top = x
        top[top>2*l] = 2*l
        top[top<0] = 0
    else:
        bot = min(max(0, x-nd*d), 2*l)
        top = max(0,min(2*l, x))
    if ret_vec:
        return bot, top, hasattr(x, '__len__')
    else:
        return bot, top

def deltaW_Area(r, r_bar, iint_fxt, t, k, l, d, v, r0, rx, deltaOnly = False, ret_G0 = False):
    bot, top = get_xrange(t,v,d,l)
    G0 = k*rx*rx*(iint_fxt(top,l) - iint_fxt(bot,l)) 
    if ret_G0:
        return G0
    thres = (r_bar/r0)*r_bar
    delta = G0*r*(r - thres)
    if deltaOnly:
        return delta
    else:
        return delta, bot, top

def deltaArea_W(half_fxt, w, uniform_w, rx, v, l, t = None, d = None, bot = None, top = None, skip_zero = False):
    if bot is None or top is None:
        assert(t is not None and d is not None)
        bot, top = get_xrange(t,v,d,l)
    pos_dArea = half_fxt(top, l)
    neg_dArea = half_fxt(bot, l)
    if uniform_w:
        dA = pos_dArea - neg_dArea
        if hasattr(w, '__len__'):
            return 2*rx*v*dA*w[0]
        else:
            return 2*rx*v*dA*w
    else:
        if len(w.shape) == 1 or (len(w.shape) == 2 and (w.shape[0] == 1 or w.shape[1] == 1)):
            nx = len(w)
            x = np.linspace(0,2*l,nx)
            w_top = np.interp(top, x, w.flatten())
            w_bot = np.interp(bot, x, w.flatten())
        else:
            nx = w.shape[0]
            n = w.shape[1]
            nt = len(top)
            x = np.linspace(0,2*l,nx)
            if n == nt:
                wt = w
            else:
                if skip_zero:
                    wt = np.array([np.interp(np.linspace(0,nt,nt+1)[1:], np.linspace(0,nt,n), w[j,:]) for j in range(nx)])
                else: # include 0
                    wt = np.array([np.interp(np.linspace(0,nt,nt), np.linspace(0,nt,n), w[j,:]) for j in range(nx)])
            w_top = np.array([np.interp(top[i], x, wt[:,i]) for i in range(nt)])
            w_bot = np.array([np.interp(bot[i], x, wt[:,i]) for i in range(nt)])
        return 2*rx*v*(pos_dArea*w_top - neg_dArea*w_bot)
