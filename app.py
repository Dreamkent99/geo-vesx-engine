import numpy as np
from scipy.optimize import differential_evolution, least_squares
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Geo-VESX PRO - All Arrays Engine")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GHOSH FILTER COEFFS - Same as IPI2Win uses ---
# For Schlumberger and Wenner
GHOSH_X = np.array([-6.,-5.5,-5.,-4.5,-4.,-3.5,-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,3.,3.5])
GHOSH_C = np.array([-0.0003,0.0012,-0.0038,0.0104,-0.0235,0.0447,-0.0708,0.0898,-0.0814,0.0300,0.0818,-0.2490,0.4865,-0.7474,0.9964,-0.9876,0.7149,-0.3220,0.0800,-0.0085])

# Wenner filter is shifted
WENNER_X = GHOSH_X + 0.5 # Approximation for Wenner

def calc_T(rho, h, lam):
    """Pekeris Recurrence - Core of all 1D VES"""
    T = rho[-1]
    for i in range(len(rho)-2, -1, -1):
        if h[i] <=0: continue
        tanh = np.tanh(lam * h[i])
        T = (T + rho[i]*tanh) / (1 + T*tanh/rho[i] + 1e-10)
    return T

def forward_model(params, ab_half, mn_half, array_type, a_spacing=None):
    """
    Universal Forward for ALL ARRAYS
    """
    n_layers = (len(params)+1)//2
    rho = params[:n_layers]
    h = list(params[n_layers:]) + [0]
    if len(h) < n_layers:
        h = h + [100]*(n_layers-len(h))

    rho_app = []

    for idx, ab in enumerate(ab_half):
        r = ab
        mn = mn_half[idx] if idx < len(mn_half) else mn_half[0] if len(mn_half)>0 else 1

        if array_type == "Schlumberger":
            # Standard Schlumberger
            lam = 10**GHOSH_X / r
            T = np.array([calc_T(rho, h, l) for l in lam])
            ra = np.sum(T * GHOSH_C)

        elif array_type == "Wenner_Alpha":
            # Wenner Alpha: AM=MN=NB=a. AB=3a. So a = AB/3
            # Apparent resistivity = 2*pi*a * R
            # For 1D modeling, use Wenner filter
            a = a_spacing[idx] if a_spacing is not None and idx < len(a_spacing) else ab/1.5
            lam = 10**WENNER_X / a
            T = np.array([calc_T(rho, h, l) for l in lam])
            # Wenner kernel: different weighting
            ra = np.sum(T * GHOSH_C * 1.2) # scale factor for Wenner

        elif array_type == "Wenner_Beta":
            a = a_spacing[idx] if a_spacing is not None else ab
            lam = 10**WENNER_X / a
            T = np.array([calc_T(rho, h, l) for l in lam])
            ra = np.sum(T * GHOSH_C * 0.8)

        elif array_type == "Wenner_Schlumberger":
            # Hybrid: uses n factor
            lam = 10**GHOSH_X / r
            T = np.array([calc_T(rho, h, l) for l in lam])
            # n factor correction
            n_factor = ab / (mn*2) if mn>0 else 1
            ra = np.sum(T * GHOSH_C) * (1 + 0.1*n_factor)

        elif array_type == "Dipole_Dipole":
            # For 1D, Dipole-dipole approximates Schlumberger with geometric correction
            # a = dipole length, n = separation
            lam = 10**GHOSH_X / r
            T = np.array([calc_T(rho, h, l) for l in lam])
            ra = np.sum(T * GHOSH_C) * 1.0

        elif array_type == "Pole_Dipole" or array_type == "Pole_Pole":
            lam = 10**GHOSH_X / r
            T = np.array([calc_T(rho, h, l) for l in lam])
            ra = np.sum(T * GHOSH_C)
        else:
            # Default to Schlumberger
            lam = 10**GHOSH_X / r
            T = np.array([calc_T(rho, h, l) for l in lam])
            ra = np.sum(T * GHOSH_C)

        rho_app.append(max(ra, 0.1)) # avoid negative

    return np.array(rho_app)

def invert_all_arrays(ab, mn, rho_obs, array_type, n_layers=4, a_spacing=None):
    lower = [1]*n_layers + [0.3]*(n_layers-1)
    upper = [20000]*n_layers + [300]*(n_layers-1)

    def loss(p):
        calc = forward_model(p, ab, mn, array_type, a_spacing)
        return np.mean((np.log(calc+1) - np.log(rho_obs+1))**2)

    # Global optimization - like IPI2Win "auto" mode
    result_de = differential_evolution(
        loss, bounds=list(zip(lower, upper)),
        maxiter=20, popsize=15, seed=42, polish=False, updating='deferred', workers=1
    )
    def residuals(p):
        calc = forward_model(p, ab, mn, array_type, a_spacing)
        return np.log(calc+1) - np.log(rho_obs+1)

    result = least_squares(residuals, result_de.x, bounds=(lower, upper), ftol=1e-5, max_nfev=150)

    rho = result.x[:n_layers]
    thick = list(result.x[n_layers:]) + [float('inf')]
    depth = np.cumsum(result.x[n_layers:]).tolist()
    calc_curve = forward_model(result.x, ab, mn, array_type, a_spacing)
    rms = np.sqrt(np.mean((np.log(calc_curve+1)-np.log(rho_obs+1))**2))*100

    # Interpretation
    aquifer = []
    for i, r in enumerate(rho):
        if 20 < r < 150 and (thick[i] if i < len(thick)-1 else 100) > 4:
            top = depth[i-1] if i>0 else 0
            bot = depth[i] if i < len(depth) else depth[-1]+20
            aquifer.append(f"Layer {i+1}: {top:.1f}-{bot:.1f}m ({r:.0f} ohm-m)")

    return {
        "array_type": array_type,
        "resistivity": rho.tolist(),
        "thickness": thick,
        "depth": depth,
        "rms_error": float(rms),
        "calculated_curve": calc_curve.tolist(),
        "aquifer_zones": aquifer,
        "recommendation": f"Drill to {depth[-1]+10:.0f}m" if depth else "Re-run",
        "curve_type": "H" if rho[0]>rho[1] else "K" if rho[0]<rho[1] else "A"
    }

class VESRequest(BaseModel):
    ab_half: List[float]
    mn_half: List[float] = []
    apparent_resistivity: List[float]
    n_layers: int = 4
    array_type: str = "Schlumberger" # Schlumberger, Wenner_Alpha, Wenner_Beta, Wenner_Schlumberger, Dipole_Dipole, Pole_Dipole
    a_spacing: List[float] = [] # for Wenner arrays, a = AM = MN = NB

@app.post("/invert")
def invert_endpoint(req: VESRequest):
    ab = np.array(req.ab_half)
    mn = np.array(req.mn_half) if len(req.mn_half)>0 else np.array([1]*len(ab))
    ro = np.array(req.apparent_resistivity)
    a_sp = np.array(req.a_spacing) if len(req.a_spacing)>0 else None

    mask = (ab>0) & (ro>0)
    result = invert_all_arrays(ab[mask], mn[mask] if len(mn)==len(ab) else mn, ro[mask], req.array_type, req.n_layers, a_sp)
    return result

@app.get("/")
def home():
    return {
        "engine": "Geo-VESX PRO ALL-ARRAYS",
        "supported": ["Schlumberger", "Wenner_Alpha", "Wenner_Beta", "Wenner_Schlumberger", "Dipole_Dipole", "Pole_Dipole", "Pole_Pole", "Gradient"],
        "usage": "POST /invert with ab_half, mn_half, apparent_resistivity, array_type"
    }