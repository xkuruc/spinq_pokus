"""Local estimation, optimization and denoising for decoded complex FID charts.

The statistical unit is one independent hardware acquisition. FID samples are
correlated observations within that acquisition, never independent shots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, hankel
from scipy.optimize import least_squares, minimize


def complex_fit(signal, sample_hz: int, *, tracked_hz: float | None = None) -> dict:
    """Fit DC + one or two damped complex modes; BIC picks the model.

    The target mode is tracked by frequency proximity across acquisitions.
    Standard errors are conditional on the selected model and noise assumptions.
    """
    y = np.asarray(signal, dtype=np.complex128)
    if len(y) < 64 or not np.all(np.isfinite(y)) or sample_hz <= 0:
        raise ValueError("FID is short, non-finite or has invalid sample frequency")
    ix = np.unique(np.linspace(0, len(y)-1, min(2048, len(y)), dtype=int))
    t = ix / sample_hz
    yy = y[ix]
    dt = 1 / sample_hz
    ft = np.fft.fftfreq(len(y), dt)
    spectrum = np.fft.fft((y-y[-max(8, len(y)//8):].mean()) * np.hanning(len(y)))
    peaks = np.argsort(np.abs(spectrum))[::-1]
    candidates = []
    for peak in peaks:
        f = float(ft[peak])
        if all(abs(f-v) > 3*sample_hz/len(y) for v in candidates):
            candidates.append(f)
        if len(candidates) >= 3:
            break
    if tracked_hz is not None:
        candidates = [float(tracked_hz)] + [f for f in candidates if abs(f-tracked_hz) > sample_hz/len(y)]
    best = None
    for order in (1, 2):
        if len(candidates) < order:
            continue
        starts = candidates[:order]
        decay = max(float(t[-1])/2, dt)
        p0 = [float(yy[-1].real), float(yy[-1].imag)]
        low, high = [-np.inf, -np.inf], [np.inf, np.inf]
        for f in starts:
            a = yy[0]-yy[-1]
            p0 += [a.real/order, a.imag/order, f, math.log(decay)]
            low += [-np.inf, -np.inf, -sample_hz/2, math.log(dt)]
            high += [np.inf, np.inf, sample_hz/2, math.log(max(100*t[-1], dt*2))]

        def model(p):
            z = np.full(len(t), p[0]+1j*p[1], dtype=np.complex128)
            for k in range(order):
                j = 2+4*k
                z += (p[j]+1j*p[j+1])*np.exp((-1/np.exp(p[j+3])+2j*np.pi*p[j+2])*t)
            return z

        def residual(p):
            e = model(p)-yy
            return np.r_[e.real, e.imag]

        fit = least_squares(residual, p0, bounds=(low, high), max_nfev=70,
                            x_scale="jac", ftol=1e-6)
        rss = float(np.sum(fit.fun**2))
        bic = len(fit.fun)*math.log(max(rss/len(fit.fun), 1e-30))+len(fit.x)*math.log(len(fit.fun))
        if best is None or bic < best[0]:
            best = (bic, fit, order, rss)
    if best is None:
        raise ValueError("No fit candidates")
    _, fit, order, rss = best
    modes = []
    cov = None
    try:
        cov = np.linalg.pinv(fit.jac.T @ fit.jac)*rss/max(1, len(fit.fun)-len(fit.x))
    except np.linalg.LinAlgError:
        pass
    for k in range(order):
        j = 2+4*k
        a = complex(fit.x[j], fit.x[j+1])
        modes.append({"amplitude": float(abs(a)), "phase_rad": float(np.angle(a)),
                      "coefficient_re": float(a.real), "coefficient_im": float(a.imag),
                      "frequency_hz": float(fit.x[j+2]), "decay_seconds": float(np.exp(fit.x[j+3])),
                      "frequency_se_hz": float(math.sqrt(max(0, cov[j+2,j+2]))) if cov is not None else None})
    target = min(modes, key=lambda m: abs(m["frequency_hz"]-tracked_hz)) if tracked_hz is not None else max(modes, key=lambda m: m["amplitude"])
    return {"target": target, "modes": modes, "model_order": order,
            "residual_rms": math.sqrt(rss/len(fit.fun)), "bic": best[0],
            "sample_hz": sample_hz, "points": len(y), "fit_points": len(ix),
            "noise_model": "complex least squares; correlated FID residuals may understate SE"}


def response_model(width_us, detuning_hz, t90_us, center_hz, linewidth_hz, offset, scale):
    rotation = np.sin(np.pi*np.asarray(width_us)/(2*t90_us))
    line = 1/(1+1j*(np.asarray(detuning_hz)-center_hz)/linewidth_hz)
    return offset+scale*rotation*line


def fit_response(observations: list[dict], *, t90_bounds=(20., 80.),
                 frequency_bounds=(-20., 20.)) -> dict:
    if len(observations) < 5:
        raise ValueError("At least five independent settings are required")
    w = np.asarray([o["width_us"] for o in observations], float)
    d = np.asarray([o["detuning_hz"] for o in observations], float)
    z = np.asarray([o["coefficient_re"]+1j*o["coefficient_im"] for o in observations])
    if len(np.unique(w)) < 3 or len(np.unique(d)) < 3:
        raise ValueError("Width and transmitter detuning each need three settings")
    sigma = np.asarray([max(float(o.get("sigma", 1)), 1e-9) for o in observations])
    init_scale = z[np.argmax(np.abs(z))]-np.mean(z)
    p0 = [np.clip(40, *t90_bounds), 0., 20., z.mean().real, z.mean().imag,
          init_scale.real, init_scale.imag]
    lower = [t90_bounds[0], frequency_bounds[0], 1., -np.inf, -np.inf, -np.inf, -np.inf]
    upper = [t90_bounds[1], frequency_bounds[1], 200., np.inf, np.inf, np.inf, np.inf]
    def residual(p):
        e=(response_model(w,d,p[0],p[1],p[2],p[3]+1j*p[4],p[5]+1j*p[6])-z)/sigma
        return np.r_[e.real,e.imag]
    fit=least_squares(residual,p0,bounds=(lower,upper),max_nfev=300)
    rss=float(np.sum(fit.fun**2))
    cov=np.linalg.pinv(fit.jac.T@fit.jac)*rss/max(1,len(fit.fun)-len(fit.x))
    return {"t90_us":float(fit.x[0]), "resonance_detuning_hz":float(fit.x[1]),
            "linewidth_hz":float(fit.x[2]), "t90_se_us":float(math.sqrt(max(cov[0,0],0))),
            "frequency_se_hz":float(math.sqrt(max(cov[1,1],0))),
            "offset_re":float(fit.x[3]), "offset_im":float(fit.x[4]),
            "scale_re":float(fit.x[5]), "scale_im":float(fit.x[6]),
            "weighted_residual_sum":rss, "points":len(observations),
            "identifiable":bool(np.linalg.matrix_rank(fit.jac)==len(fit.x) and
                                t90_bounds[0]+1<fit.x[0]<t90_bounds[1]-1 and
                                frequency_bounds[0]+1<fit.x[1]<frequency_bounds[1]-1)}


class ComplexGridBayes:
    """Profiled-nuisance posterior over resonance and Rabi 90-degree width."""
    def __init__(self, t90_range=(25., 65.), frequency_range=(-20.,20.)):
        self.t90=np.linspace(*t90_range, 41)
        self.freq=np.linspace(*frequency_range, 41)
        self.observations=[]
        self.posterior=np.full((len(self.t90),len(self.freq)),1/(len(self.t90)*len(self.freq)))
        self.map_nuisance=(0j,1+0j)

    def update(self, observations: list[dict], linewidth_hz: float):
        self.observations=list(observations)
        if len(observations)<2:
            return
        w=np.array([o["width_us"] for o in observations]); d=np.array([o["detuning_hz"] for o in observations])
        y=np.array([complex(o["coefficient_re"],o["coefficient_im"]) for o in observations])
        sigma=np.array([max(float(o.get("sigma",1)),1e-9) for o in observations])
        scores=np.empty_like(self.posterior)
        nuisances={}
        for i,t90 in enumerate(self.t90):
            for j,f0 in enumerate(self.freq):
                g=response_model(w,d,t90,f0,linewidth_hz,0j,1+0j)
                design=np.column_stack((np.ones(len(w)),g))/sigma[:,None]
                pars=np.linalg.lstsq(design,y/sigma,rcond=None)[0]
                scores[i,j]=-float(np.sum(np.abs((design@pars-y/sigma))**2))/2
                nuisances[i,j]=pars
        scores-=scores.max()
        prob=np.exp(scores)
        self.posterior=prob/prob.sum()
        index=np.unravel_index(np.argmax(prob),prob.shape)
        self.map_nuisance=tuple(nuisances[index])

    def summary(self):
        t,f=np.meshgrid(self.t90,self.freq,indexing="ij")
        mt=float(np.sum(self.posterior*t)); mf=float(np.sum(self.posterior*f))
        return {"t90_us":mt,"resonance_detuning_hz":mf,
                "t90_sd_us":float(np.sqrt(np.sum(self.posterior*(t-mt)**2))),
                "frequency_sd_hz":float(np.sqrt(np.sum(self.posterior*(f-mf)**2)))}

    def next_setting(self, candidates, linewidth_hz: float, sigma: float):
        o,s=self.map_nuisance
        best=None
        for w,d in candidates:
            if any(abs(w-x["width_us"])<.01 and abs(d-x["detuning_hz"])<.01 for x in self.observations):
                continue
            t,f=np.meshgrid(self.t90,self.freq,indexing="ij")
            pred=response_model(w,d,t,f,linewidth_hz,o,s)
            mean=np.sum(self.posterior*pred)
            var=float(np.sum(self.posterior*np.abs(pred-mean)**2))
            # One complex datum; correlated points in its FID are already reduced to one fit.
            gain=math.log1p(var/max(sigma*sigma,1e-12))
            if best is None or gain>best[0]: best=(gain,(w,d))
        return None if best is None else best[1]


def choose_fid_budget(pilot: list[dict], lengths=(4000,8000,16000), target_se_hz=2.0):
    """Choose the cheapest *real* sampleCount/repetition pair from pilot fits."""
    rows=[]
    for n in lengths:
        matches=[p for p in pilot if p["sample_count"]==n]
        if len(matches)<2: continue
        vals=np.array([p["frequency_hz"] for p in matches])
        sd=float(np.std(vals,ddof=1))
        for repeats in (1,2,3,4):
            se=sd/math.sqrt(repeats)
            rows.append({"sample_count":n,"repeats":repeats,"predicted_se_hz":se,
                         "sample_budget":n*repeats})
    feasible=[x for x in rows if x["predicted_se_hz"]<=target_se_hz]
    return min(feasible or rows,key=lambda x:x["sample_budget"]) if rows else None


def choose_echo_time(times, amplitudes, candidate_times, noise_sd):
    """D-optimal next echo time for log A - t/T2 model, for future capable SDKs."""
    if len(times)<3 or any(a<=0 for a in amplitudes): return None
    t=np.asarray(times,float); a=np.asarray(amplitudes,float)
    fit=np.polyfit(t,np.log(a),1)
    if fit[0]>=0:return None
    variance=max((noise_sd/np.maximum(a,1e-9))**2)
    fisher=np.column_stack((np.ones(len(t)),t)).T@np.column_stack((np.ones(len(t)),t))/variance
    return max(candidate_times,key=lambda x:np.linalg.slogdet(fisher+np.outer([1,x],[1,x])/variance)[1])


def hankel_denoise(signal, rank=2, max_matrix=128):
    """Overlapped low-rank Hankel SSA at the original sampling rate.

    No decimation is used, so high-frequency NMR components cannot alias merely
    because the matrix size was bounded for memory.
    """
    y=np.asarray(signal,complex)
    if len(y)<32: raise ValueError("FID too short")
    window=min(1023,len(y));hop=max(1,window//2)
    starts=list(range(0,max(1,len(y)-window+1),hop))
    last=len(y)-window
    if starts[-1]!=last:starts.append(last)
    output=np.zeros(len(y),complex);weights=np.zeros(len(y))
    for start in starts:
        chunk=y[start:start+window]
        rows=min(max_matrix,max(16,len(chunk)//3))
        h=hankel(chunk[:rows],chunk[rows-1:])
        u,s,vh=np.linalg.svd(h,full_matrices=False)
        clean=(u[:,:rank]*s[:rank])@vh[:rank]
        sums=np.zeros(len(chunk),complex);counts=np.zeros(len(chunk))
        for i in range(clean.shape[0]):
            sums[i:i+clean.shape[1]]+=clean[i]
            counts[i:i+clean.shape[1]]+=1
        segment=sums/counts
        taper=np.maximum(np.hanning(len(chunk)),.05)
        output[start:start+window]+=segment*taper
        weights[start:start+window]+=taper
    return output/weights


def rotation(pulses, *, t90_us, reference_amplitude_pct, detuning_hz=0., amplitude_scale=1.):
    """Single uncoupled H-spin design model; not a measured hardware fidelity."""
    sx=np.array([[0,1],[1,0]],complex); sy=np.array([[0,-1j],[1j,0]],complex)
    sz=np.diag([1.,-1.]).astype(complex)
    u=np.eye(2,dtype=complex)
    omega90=np.pi/(2*t90_us*1e-6)
    for p in pulses:
        dur=p["width"]*1e-6
        omega=omega90*p["am"]/reference_amplitude_pct*amplitude_scale
        phase=np.deg2rad(p["phase"])
        h=.5*(omega*(math.cos(phase)*sx+math.sin(phase)*sy)+2*np.pi*(detuning_hz+p["freshift"])*sz)
        u=expm(-1j*h*dur)@u
    return u


def bb1_pulses(t90_us, phase_deg, amplitude_pct):
    """Wimperis BB1: target theta, 180_phi, 360_3phi, 180_phi."""
    # theta=pi/2, hence phi=acos(-theta/(4*pi))=acos(-1/8).
    phi=math.degrees(math.acos(-.125))
    return [{"width":w,"am":amplitude_pct,"phase":(phase_deg+p)%360,"freshift":0.}
            for w,p in ((t90_us,0),(2*t90_us,phi),(4*t90_us,3*phi),(2*t90_us,phi))]


def optimize_grape(t90_us, reference_amplitude_pct, *, phase_deg=90., segments=4,
                   max_total_us=200., seed=42):
    """Bounded segmented local robust design via L-BFGS-B numerical gradient."""
    if not 15<t90_us<80 or not 0<reference_amplitude_pct<=100: raise ValueError("unverified calibration")
    duration=min(max_total_us,2*t90_us)
    width=duration/segments
    if width<5: raise ValueError("segment resolution below study minimum")
    target=rotation([{"width":t90_us,"am":reference_amplitude_pct,"phase":phase_deg,"freshift":0.}],
                    t90_us=t90_us,reference_amplitude_pct=reference_amplitude_pct)
    design_grid=[(a,d) for a in (.95,1.,1.05) for d in (-10.,0.,10.)]
    def decode(x):
        return [{"width":width,"am":float(x[i]),"phase":float(x[i+segments])%360,"freshift":0.}
                for i in range(segments)]
    def loss(x):
        pulses=decode(x)
        errors=[]
        for scale,delta in design_grid:
            u=rotation(pulses,t90_us=t90_us,reference_amplitude_pct=reference_amplitude_pct,
                       amplitude_scale=scale,detuning_hz=delta)
            errors.append(1-abs(np.trace(target.conj().T@u))/2)
        return float(np.mean(errors)+.5*max(errors)+.0001*sum(p["am"]**2*p["width"] for p in pulses)/10000)
    x0=np.r_[np.full(segments,reference_amplitude_pct*t90_us/duration),np.full(segments,phase_deg)]
    rng=np.random.default_rng(seed)
    fits=[]
    for start in (x0,np.r_[np.clip(x0[:segments]*(1+rng.normal(0,.1,segments)),1,reference_amplitude_pct),
                            np.mod(x0[segments:]+rng.normal(0,15,segments),360)]):
        fit=minimize(loss,start,method="L-BFGS-B",bounds=[(1,reference_amplitude_pct)]*segments+[(0,360)]*segments,
                     options={"maxiter":60,"maxfun":600})
        fits.append(fit)
    best=min(fits,key=lambda x:x.fun)
    return {"pulses":decode(best.x),"simulated_design_loss":float(best.fun),
            "design_grid":design_grid,"target":"H 90-degree rotation proxy",
            "model":"uncoupled single spin; H-P coupling unmeasured; hardware test required",
            "total_width_us":duration,"relative_rf_cost":float(sum(p["am"]**2*p["width"] for p in decode(best.x))/(reference_amplitude_pct**2*t90_us))}


def bounded_nelder_mead(objective, start, bounds, budget):
    """True simplex updates with a strict hardware evaluation budget."""
    history=[]
    seen={}
    def evaluate(x):
        key=tuple(round(float(v),4) for v in x)
        if key in seen:return seen[key]
        if len(history)>=budget: raise StopIteration
        if any(v<lo or v>hi for v,(lo,hi) in zip(x,bounds)):
            return 1e12+sum(max(lo-v,0,v-hi)**2 for v,(lo,hi) in zip(x,bounds))
        value=float(objective(np.asarray(x,float)))
        history.append({"x":list(map(float,x)),"loss":value})
        seen[key]=value
        return value
    try:
        minimize(evaluate,np.asarray(start,float),method="Nelder-Mead",
                 options={"maxfev":budget,"xatol":.1,"fatol":.01,"adaptive":True})
    except StopIteration:
        pass
    return min(history,key=lambda x:x["loss"]),history


def gaussian_process_optimize(objective, start, bounds, budget, seed=42):
    """Bounded GP/EI minimization, with one real evaluation per candidate."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
    rng=np.random.default_rng(seed)
    lows=np.array([x[0] for x in bounds]); widths=np.array([x[1]-x[0] for x in bounds])
    history=[]
    candidates=[np.asarray(start,float)]
    while len(history)<budget:
        if candidates:
            x=candidates.pop(0)
        else:
            xtrain=np.array([h["x"] for h in history]); ytrain=np.array([h["loss"] for h in history])
            xn=(xtrain-lows)/widths
            kernel=ConstantKernel(1.,(1e-3,1e3))*Matern(length_scale=np.ones(len(bounds)),nu=2.5)+WhiteKernel(noise_level=1e-3)
            gp=GaussianProcessRegressor(kernel=kernel,normalize_y=True,random_state=seed,
                                         n_restarts_optimizer=1)
            gp.fit(xn,ytrain)
            pool=rng.uniform(0,1,(256,len(bounds)))
            pool=np.vstack((pool,np.clip(xn[np.argmin(ytrain)]+rng.normal(0,.08,(64,len(bounds))),0,1)))
            mu,sd=gp.predict(pool,return_std=True)
            improvement=ytrain.min()-mu-.01
            z=improvement/np.maximum(sd,1e-9)
            ei=improvement*norm.cdf(z)+sd*norm.pdf(z)
            x=lows+pool[np.argmax(ei)]*widths
        loss=float(objective(x))
        history.append({"x":list(map(float,x)),"loss":loss})
    return min(history,key=lambda h:h["loss"]),history
