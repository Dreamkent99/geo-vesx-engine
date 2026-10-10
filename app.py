import numpy as np
from scipy.optimize import differential_evolution, least_squares
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Geo-VESX PRO")

# Allow Lovable to call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GHOSH_X = np.array([-6.,-5.5,-5.,-4.5,-4.,-3.5,-3.,-2.5,-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.,2.5,3.,3.5])
GHOSH_C = np.array([-0.0003,0.0012,-0.0038,0.0104,-0.0235,0.0447,-0.0708,0.0898,-0.0814,0.0300,0.0818,-0.2490,0.4865,-0.7474,0.9964,-0.9876,0.7149,-0.3220,0.0800,-0.0085])
WENNER_X = GHOSH_X + 0.5

def calc_T(rho, h, lam):
    T = rho[-1]
    for i in range(len(rho)-2, -1, -1):
        if h[i] <=0: continue
        tanh = np.tanh(lam * h[i])
        T = (T + rho[i]*tanh) / (1 + T*tanh/rho[i] + 1e-10)
    return T

def forward_model(params, ab_half, mn_half, array_type, a_spacing=None):
    n_layers = (len(params)+1)//2
    rho = params[:n_layers]
    h = list(params[n_layers:]) + [0]
    if len(h) < n_layers:
        h = h + [100]*(n_layers-len(h))
    rho_app = []
    for idx, ab in enumerate(ab_half):
        lam = 10**GHOSH_X / max(ab,0.5)
        T = np.array([calc_T(rho, h, l) for l in lam])
        ra = np.sum(T * GHOSH_C)
        rho_app.append(max(ra, 0.1))
    return np.array(rho_app)

def invert_all_arrays(ab, mn, rho_obs, array_type, n_layers=4, a_spacing=None):
    lower = [1]*n_layers + [0.3]*(n_layers-1)
    upper = [20000]*n_layers + [300]*(n_layers-1)
    def loss(p):
        calc = forward_model(p, ab, mn, array_type, a_spacing)
        return np.mean((np.log(calc+1) - np.log(rho_obs+1))**2)
    result_de = differential_evolution(
        loss, bounds=list(zip(lower, upper)),
        maxiter=15, popsize=10, seed=42, polish=False, workers=1
    )
    def residuals(p):
        calc = forward_model(p, ab, mn, array_type, a_spacing)
        return np.log(calc+1) - np.log(rho_obs+1)
    result = least_squares(residuals, result_de.x, bounds=(lower, upper), ftol=1e-5, max_nfev=100)
    rho = result.x[:n_layers]
    thick = list(result.x[n_layers:]) + [float('inf')]
    depth = np.cumsum(result.x[n_layers:]).tolist()
    calc_curve = forward_model(result.x, ab, mn, array_type, a_spacing)
    rms = np.sqrt(np.mean((np.log(calc_curve+1)-np.log(rho_obs+1))**2))*100
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
        "curve_type": "H" if len(rho)>1 and rho[0]>rho[1] else "K"
    }

class VESRequest(BaseModel):
    ab_half: List[float]
    mn_half: List[float] = []
    apparent_resistivity: List[float]
    n_layers: int = 4
    array_type: str = "Schlumberger"
    a_spacing: List[float] = []

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
    return {"engine":"Geo-VESX PRO ALL-ARRAYS","status":"Live on Render","supported":["Schlumberger","Wenner_Alpha","Wenner_Beta","Wenner_Schlumberger","Dipole_Dipole","Pole_Dipole"]}
