"""
Gradientes Conjugados Precondicionado (GCP) aplicado a finanzas
Caso: suavizador de presupuesto anual · Ingeniería de Sistemas · UMSA

Algoritmo 2 de: Suñagua S., P. (2020). Método de Gradientes Conjugados
Precondicionado. Revista Boliviana de Matemática – UMSA, 04, 2–7.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ─────────────────────────── Datos del estudiante (edítalos) ───────────────────────────
ESTUDIANTE = "Valeria Isabel Estrugo Pérez"
MENCION = "Ingeniería de Sistemas"
ASIGNATURA = "Métodos Numéricos"
UNIVERSIDAD = "Universidad Mayor de San Andrés · Carrera de Informática"

st.set_page_config(page_title="GCP · Suavizador de presupuesto", page_icon=":material/query_stats:", layout="wide")
S = st.session_state

# ─────────────────────────────────── Modelo financiero ───────────────────────────────────
N = 12
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
# Presupuesto deseado por mes (miles de Bs). Datos ilustrativos con estacionalidad típica:
# gasto escolar/inicio de gestión al comienzo, aguinaldo y fiestas de fin de año al final.
# Reemplázalos por los de tu empresa si tienes una fuente real.
P_BASE = np.array([80, 80, 90, 90, 100, 100, 100, 110, 110, 120, 130, 160], float)
P_ESTRES = P_BASE * np.array([1] * 10 + [1.4, 1.4])


def T_mat(n=N):
    T = 2 * np.eye(n) - np.eye(n, k=1) - np.eye(n, k=-1)
    T[0, 0] = T[-1, -1] = 1
    return T


def build_A(lam, n=N):
    return np.eye(n) + lam * T_mat(n)


SC = {
    "Ideal": dict(
        lam=0.5, b=P_BASE,
        why="Política de suavizado moderada (λ = 0.5). El presupuesto sigue la estacionalidad y solo recorta los picos más agudos.",
    ),
    "Bajo estrés": dict(
        lam=0.5, b=P_ESTRES,
        why="Aguinaldo y fiestas de fin de año presionan noviembre y diciembre (+40 % en b). La estructura A no cambia: solo se desplaza el punto de equilibrio.",
    ),
    "Mal condicionado": dict(
        lam=200.0, b=P_BASE,
        why="Una cláusula tipo covenant exige gasto casi constante (λ = 200). Los meses quedan fuertemente acoplados fuera de la diagonal y κ(A) se dispara.",
    ),
}

# ──────────────────────────────── Núcleo numérico (propio) ────────────────────────────────


def cholesky(A):
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError(f"pivote no positivo en la fila {i + 1}")
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def fsub(L, b):
    y = np.zeros(len(b))
    for i in range(len(b)):
        y[i] = (b[i] - L[i, :i] @ y[:i]) / L[i, i]
    return y


def bsub(U, b):
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i + 1:] @ x[i + 1:]) / U[i, i]
    return x


def ic0(A):
    """Cholesky incompleto IC(0): conserva solo el patrón de ceros de A."""
    n = len(A)
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if A[i, j] == 0:
                continue
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("IC(0) falló: pivote no positivo")
                L[i, i] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    return L


def make_precond(A, kind, omega=1.0):
    """Devuelve (solve, M): solve(r) resuelve M z = r."""
    d = np.diag(A)
    if kind == "Jacobi":
        return (lambda r: r / d), np.diag(d)
    if kind == "IC(0)":
        L = ic0(A)
        return (lambda r: bsub(L.T, fsub(L, r))), L @ L.T
    if kind == "SSOR":
        Dw = np.diag(d) / omega
        K = Dw + np.tril(A, -1)
        c = omega / (2 - omega)
        M = c * K @ np.diag(omega / d) @ K.T
        return (lambda r: (1 / c) * bsub(K.T, Dw @ fsub(K, r))), M
    return (lambda r: r.copy()), np.eye(len(A))


def pcg(A, b, x0, solve, tol, nmax):
    """Algoritmo 2 (Suñagua, 2020): resuelve M z_k = r_k en cada iteración."""
    x = x0.copy()
    r = b - A @ x
    nb = np.linalg.norm(b)
    hist = [np.linalg.norm(r) / nb]
    p, rz_old, k = None, None, 0
    while hist[-1] >= tol and k < nmax:
        z = solve(r)
        rz = r @ z
        p = z.copy() if k == 0 else z + (rz / rz_old) * p
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rz_old = rz
        k += 1
        hist.append(np.linalg.norm(r) / nb)
    return dict(x=x, k=k, hist=np.array(hist), ok=bool(hist[-1] < tol))


def cond_spd(A):
    e = np.linalg.eigvalsh(A)
    return e[-1] / e[0]


def cond_prec(A, M):
    """κ(M⁻¹A) vía la matriz simétrica equivalente L⁻¹ A L⁻ᵀ, con M = L Lᵀ."""
    Lm = cholesky(M)
    Z = np.linalg.solve(Lm, np.linalg.solve(Lm, A).T)
    e = np.linalg.eigvalsh((Z + Z.T) / 2)
    return e[-1] / e[0]


def bound_iters(kappa, tol):
    """Menor k tal que 2((√κ−1)/(√κ+1))^k ≤ tol  (ec. 1 del artículo)."""
    s = np.sqrt(kappa)
    rho = (s - 1) / (s + 1)
    return 1 if rho < 1e-12 else int(np.ceil(np.log(tol / 2) / np.log(rho)))


def validate(A, b, tol_sym=1e-9):
    errs = []
    A, b = np.asarray(A, float), np.asarray(b, float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return ["A debe ser cuadrada (n×n)."]
    if A.shape[0] < 4:
        errs.append("Se requiere n ≥ 4.")
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return errs + ["Hay celdas vacías o no numéricas en A o b."]
    if A.shape[0] != b.size:
        return errs + ["b debe tener la misma dimensión que A."]
    D = np.abs(A - A.T)
    if D.max() > tol_sym * max(1.0, np.abs(A).max()):
        i, j = np.unravel_index(np.argmax(D), D.shape)
        errs.append(f"A no es simétrica: |a[{i + 1},{j + 1}] − a[{j + 1},{i + 1}]| = {D.max():.3g}.")
    else:
        try:
            cholesky(A)
        except ValueError as e:
            errs.append(f"A no es definida positiva ({e}).")
    if np.linalg.norm(b) == 0:
        errs.append("‖b‖₂ = 0: el residuo relativo no está definido.")
    return errs


def run_all(A, b, x0, tol, nmax, kind, omega=1.0):
    L = cholesky(A)
    xref = bsub(L.T, fsub(L, b))
    gc = pcg(A, b, x0, lambda r: r.copy(), tol, nmax)
    note = ""
    try:
        solve, M = make_precond(A, kind, omega)
    except ValueError as e:
        solve, M = make_precond(A, "Jacobi")
        note = f"{e}. Se usó Jacobi como alternativa."
        kind = "Jacobi"
    gcp = pcg(A, b, x0, solve, tol, nmax)
    nb = np.linalg.norm(b)
    return dict(
        A=A, b=b, L=L, xref=xref, gc=gc, gcp=gcp, kind=kind, note=note,
        kA=cond_spd(A), kM=cond_prec(A, M), tol=tol,
        res_gc=np.linalg.norm(b - A @ gc["x"]) / nb, res_gcp=np.linalg.norm(b - A @ gcp["x"]) / nb,
    )


def error_table(xref, x):
    ea = np.abs(xref - x)
    with np.errstate(divide="ignore", invalid="ignore"):
        er = np.where(xref != 0, ea / np.abs(xref), np.nan)
    names = [f"x{i + 1} · {MESES[i]}" for i in range(len(xref))] if len(xref) == N else [f"x{i + 1}" for i in range(len(xref))]
    return pd.DataFrame({"Variable": names, "x_ref": xref, "x_GCP": x, "Ea": ea, "Er": er, "Ep (%)": er * 100})


# ───────────────────────────────────── Estado global ─────────────────────────────────────


def load_scenario(name):
    sc = SC[name]
    S.A, S.b, S.x0 = build_A(sc["lam"]), sc["b"].copy(), np.zeros(N)
    S.scen, S.ver = name, S.get("ver", 0) + 1


def init():
    if "A" not in S:
        load_scenario("Ideal")
    for k, v in dict(tol=1e-6, nmax=200, kind="IC(0)", omega=1.2, autosym=True, x0custom=False).items():
        S.setdefault(k, v)


def x0_used():
    return S.x0.astype(float) if S.x0custom else np.zeros(len(S.b))


def current():
    errs = validate(S.A, S.b)
    if errs:
        return None, errs
    return run_all(S.A, S.b, x0_used(), S.tol, S.nmax, S.kind, S.omega), []


def scen_results():
    return {n: run_all(build_A(sc["lam"]), sc["b"], np.zeros(N), S.tol, S.nmax, S.kind, S.omega) for n, sc in SC.items()}


# ─────────────────────────────────────── Estilo ───────────────────────────────────────

CSS = """
<style>
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp td, .stApp th, .stApp input, .stApp textarea, .stApp button, .stApp [data-baseweb="select"] div {
  font-family: Consolas, "Cascadia Mono", "SF Mono", Menlo, "DejaVu Sans Mono", monospace;
}
.block-container { padding-top: 2.2rem; max-width: 1180px; }
.hero { padding: 2rem 2.2rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.30);
  background: linear-gradient(120deg, rgba(13,148,136,.22), rgba(124,58,237,.18)); margin-bottom: 1.2rem; }
.hero h1 { margin: 0 0 .5rem 0; font-size: 2.1rem; line-height: 1.15; letter-spacing: -.5px; padding: 0; }
.hero p { margin: .2rem 0; opacity: .9; }
.chip { display: inline-block; padding: .12rem .65rem; border-radius: 999px; font-size: .8rem; margin: .5rem .4rem 0 0;
  border: 1px solid rgba(13,148,136,.7); color: #0d9488; font-weight: 600; }
.sec { margin-bottom: .6rem; padding-left: .9rem; border-left: 4px solid #0d9488; }
.sec h2 { margin: 0; padding: 0; font-size: 1.6rem; }
.sec p { margin: .2rem 0 0 0; opacity: .75; }
.kpi { border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .8rem 1rem; background: rgba(128,128,128,.07); }
.kpi .v { font-size: 1.55rem; font-weight: 700; color: #0d9488; line-height: 1.2; }
.kpi .l { font-size: .8rem; opacity: .75; }
.card { border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .9rem 1.1rem; background: rgba(128,128,128,.06); height: 100%; }
.card b { color: #7c3aed; }
.ok { color: #059669; font-weight: 700; } .bad { color: #e11d48; font-weight: 700; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.25); }
[data-testid="stSidebarNavLink"] { border-radius: 10px; margin-bottom: 2px; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(13,148,136,.18); border-left: 3px solid #0d9488; font-weight: 700; }
.stButton > button { border-radius: 10px; }
</style>
"""


def hdr(title, sub):
    st.markdown(f"<div class='sec'><h2>{title}</h2><p>{sub}</p></div>", unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(f"<div class='kpi'><div class='v'>{value}</div><div class='l'>{label}</div></div>", unsafe_allow_html=True)


def style(fig, h=360):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                      colorway=["#0d9488", "#7c3aed", "#f59e0b", "#e11d48"], font=dict(family="Consolas, Menlo, monospace"))
    return fig


def vec(v, f="{:.2f}"):
    return "[" + ", ".join(f.format(t) for t in v) + "]"


def blocked(errs):
    st.error("Los datos actuales no permiten resolver el sistema:\n\n" + "\n".join(f"- {e}" for e in errs))
    st.info("Corrígelos en «Entrada de datos» o carga un escenario en «Tres escenarios».")


# ───────────────────────────────────────── Páginas ─────────────────────────────────────────


def p_inicio():
    st.markdown(
        f"""<div class='hero'><h1>Suavizador de presupuesto anual con Gradientes Conjugados Precondicionado</h1>
        <p>Cómo repartir un gasto de 12 meses sin saltos bruscos, resuelto como un sistema simétrico definido positivo Ax = b.</p>
        <span class='chip'>{MENCION}</span><span class='chip'>{ASIGNATURA}</span><span class='chip'>n = 12 variables</span><span class='chip'>tolerancia 10⁻⁶</span>
        <p style='margin-top:1rem'><b>{ESTUDIANTE}</b><br>{UNIVERSIDAD}</p></div>""",
        unsafe_allow_html=True,
    )
    R = scen_results()
    c = st.columns(3)
    for col, (n, r) in zip(c, R.items()):
        kpi(col, f"{n}: κ(A) → κ(M⁻¹A) con {r['kind']}", f"{r['kA']:.3g} → {r['kM']:.3g}")
    st.markdown("&nbsp;")
    a, b = st.columns(2)
    a.markdown("<div class='card'><b>El problema.</b> Una empresa necesita repartir su presupuesto del año entre 12 meses. "
               "Gastar justo lo que pide cada mes provoca picos (aguinaldo, fin de gestión) que desordenan el flujo de caja. "
               "El modelo equilibra dos fuerzas: acercarse al presupuesto deseado y evitar saltos entre meses consecutivos.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><b>El método.</b> El modelo lleva a (I + λT)x = p, una matriz simétrica definida positiva. "
               "Se resuelve con Cholesky (referencia) y con Gradientes Conjugados con y sin precondicionador, siguiendo el Algoritmo 2 de Suñagua (2020). "
               "Usa el menú de la izquierda para recorrer el trabajo.</div>", unsafe_allow_html=True)


def p_caso():
    hdr("Caso de estudio", "Contexto financiero, justificación del modelo y procedencia de los datos")
    st.markdown(
        """
**Contexto.** Una empresa de software mediana en La Paz aprueba su presupuesto operativo anual y debe distribuirlo mes a mes.
Las necesidades reales son estacionales: gasto de inicio de gestión en el primer trimestre, y presión fuerte en noviembre y diciembre por aguinaldo y cierre de año.
Si el gasto asignado sigue esos picos sin control, el flujo de caja se desordena y se tensiona la liquidez.

**Justificación del modelo.** El gerente financiero busca el plan de gasto *x* que minimice

"""
    )
    st.latex(r"f(x)=\tfrac12\,\|x-p\|_2^2+\tfrac{\lambda}{2}\sum_{i=1}^{n-1}(x_{i+1}-x_i)^2")
    st.markdown(
        """
El primer término mide cuánto se aleja el plan del presupuesto deseado *p*. El segundo penaliza los saltos entre meses consecutivos, y **λ** es la rigidez de la política de suavizado.
Como *f* es cuadrática y convexa, su mínimo satisface ∇f = 0, que es el sistema lineal **(I + λT)x = p**.

**Procedencia de los datos.** El vector *p* es ilustrativo, construido con la estacionalidad típica descrita arriba, en miles de Bs. Si cuentas con datos reales de tu empresa,
cámbialos en «Entrada de datos»: la web recalcula todo. Los valores de λ de cada escenario son parámetros de política, no mediciones.

**Mención: Ingeniería de Sistemas.** El trabajo modela un proceso de negocio completo (datos → validación → solución → control de reglas), con reglas de negocio y flujo de caja acumulado en la sección «Solución de referencia».
"""
    )
    st.graphviz_chart(
        """digraph { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#99f6e4", color="#0d9488", fontcolor="#042f2e", fontname="Consolas"]; edge [color="#7c3aed"];
        "Presupuesto deseado p" -> "Validación SPD" -> "Cholesky (referencia)" -> "GC vs GCP" -> "Errores y κ" -> "Reglas de negocio"; }""",
        width="stretch",
    )


def spd_row(name, A):
    D = np.abs(A - A.T).max()
    e = np.linalg.eigvalsh(A)
    minors = [np.linalg.slogdet(A[:k, :k]) for k in range(1, len(A) + 1)]
    mm = min(s * np.exp(l) for s, l in minors)
    return {"Escenario": name, "max|A−Aᵀ|": f"{D:.1e}", "λ_min": f"{e[0]:.4g}", "λ_max": f"{e[-1]:.4g}", "κ(A)": f"{e[-1] / e[0]:.4g}",
            "menor principal mínimo": f"{mm:.3e}", "SPD": "✓" if D < 1e-12 and e[0] > 0 and mm > 0 else "✗"}


def p_formulacion():
    hdr("Formulación", "Variables, sistema Ax = b, interpretación de coeficientes y verificación SPD")
    st.markdown("**Variables (n = 12)**, todas en miles de Bs:")
    st.dataframe(pd.DataFrame({"Variable": [f"x{i + 1}" for i in range(N)], "Significado": [f"Gasto asignado al mes de {m}" for m in MESES]}),
                 hide_index=True, width="stretch", height=250)
    st.latex(r"A=I+\lambda T,\quad T=\begin{pmatrix}1&-1&&\\-1&2&-1&\\&\ddots&\ddots&\ddots\\&&-1&1\end{pmatrix},\qquad b=p")
    a, b = st.columns(2)
    a.markdown("<div class='card'><b>Interpretación de aᵢⱼ.</b><br>Diagonal aᵢᵢ = 1 + 2λ (1 + λ en enero y diciembre): fuerza con que se respeta el deseo del mes más su acople con vecinos.<br>"
               "Fuera de la diagonal aᵢ,ᵢ₊₁ = −λ: costo de «saltar» entre meses consecutivos. El resto de coeficientes es 0.</div>", unsafe_allow_html=True)
    b.markdown("<div class='card'><b>Propiedades.</b><br>xᵀAx = ‖x‖² + λΣ(xᵢ₊₁ − xᵢ)² &gt; 0 para todo x ≠ 0: A es simétrica y definida positiva si λ ≥ 0.<br>"
               "Como 1ᵀT = 0, se cumple Σxᵢ = Σpᵢ: el suavizado redistribuye el presupuesto sin crearlo ni destruirlo.</div>", unsafe_allow_html=True)
    st.markdown("#### Verificación formal en los tres escenarios")
    st.dataframe(pd.DataFrame([spd_row(n, build_A(sc["lam"])) for n, sc in SC.items()]), hide_index=True, width="stretch")
    st.caption("Criterios: A = Aᵀ, autovalores positivos y criterio de Sylvester (todos los menores principales positivos). κ(A) = λ_max/λ_min = 1 + 2λ(1 − cos((n−1)π/n)).")
    st.markdown("#### Matriz A del escenario cargado")
    fig = go.Figure(go.Heatmap(z=S.A, x=[f"x{j + 1}" for j in range(len(S.A))], y=[f"x{i + 1}" for i in range(len(S.A))], zmid=0, zsmooth=False,
                               colorscale=[[0, "#7c3aed"], [.5, "rgba(128,128,128,.15)"], [1, "#0d9488"]]))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 420), width="stretch")
    with st.expander("Menores principales de la matriz actual"):
        mp = [np.linalg.slogdet(S.A[:k, :k]) for k in range(1, len(S.A) + 1)]
        st.dataframe(pd.DataFrame({"k": range(1, len(mp) + 1), "det(A[1:k,1:k])": [f"{s * np.exp(l):.4e}" for s, l in mp]}), hide_index=True)


def p_entrada():
    hdr("Entrada de datos", "Edita A y b, elige x₀, tolerancia y precondicionador. Todo se recalcula al instante")
    st.markdown(f"Escenario cargado: **{S.scen}**")
    c1, c2, c3, c4 = st.columns(4)
    S.tol = c1.selectbox("Tolerancia ‖r‖/‖b‖", [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10], index=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10].index(S.tol), format_func=lambda v: f"{v:.0e}")
    S.nmax = int(c2.number_input("N_máx de iteraciones", 1, 5000, int(S.nmax)))
    S.kind = c3.selectbox("Precondicionador (GCP)", ["Jacobi", "IC(0)", "SSOR"], index=["Jacobi", "IC(0)", "SSOR"].index(S.kind))
    if S.kind == "SSOR":
        S.omega = c4.slider("ω (SSOR)", 0.1, 1.9, float(S.omega), 0.05)
    else:
        c4.markdown("<small>Jacobi sirve para mal escalamiento diagonal. IC(0) y SSOR para acople fuerte fuera de la diagonal.</small>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    S.autosym = d1.toggle("Simetría automática", S.autosym, help="Al editar aᵢⱼ también se escribe aⱼᵢ.")
    S.x0custom = d2.toggle("x₀ personalizado", S.x0custom, help="Por defecto x₀ = 0.")
    if d3.button("Simetrizar (A+Aᵀ)/2", width="stretch"):
        S.A = (S.A + S.A.T) / 2
        S.ver += 1
        st.rerun()
    if d4.button("Restablecer escenario", width="stretch"):
        load_scenario(S.scen)
        st.rerun()
    e1, e2 = st.columns([1, 2])
    lam = e1.number_input("Reconstruir A = I + λT con λ =", 0.0, 10000.0, 0.5, 0.5)
    if e2.button("Aplicar λ", help="Reemplaza A por I + λT (b no cambia)."):
        S.A = build_A(lam)
        S.ver += 1
        st.rerun()

    n = len(S.b)
    st.markdown("**Matriz A** (celdas editables)")
    idx = [f"x{i + 1}" for i in range(n)]
    ed = st.data_editor(pd.DataFrame(S.A, columns=idx, index=idx), key=f"edA{S.ver}", width="stretch")
    newA = ed.to_numpy(float)
    changed = ~np.isclose(newA, S.A, rtol=0, atol=0, equal_nan=True)
    if changed.any():
        if S.autosym:
            for i, j in np.argwhere(changed):
                newA[j, i] = newA[i, j]
            S.A = newA
            S.ver += 1
            st.rerun()
        S.A = newA

    st.markdown("**Vector b (presupuesto deseado, miles de Bs) y x₀**")
    edb = st.data_editor(pd.DataFrame({"Mes": MESES, "b": S.b, "x0": S.x0}), key=f"edb{S.ver}", hide_index=True, width="stretch",
                         disabled=["Mes"], column_config={"b": st.column_config.NumberColumn("b (miles Bs)", format="%.4f"), "x0": st.column_config.NumberColumn("x₀", format="%.4f")})
    S.b, S.x0 = edb["b"].to_numpy(float), edb["x0"].to_numpy(float)
    errs = validate(S.A, S.b)
    if errs:
        st.error("Datos no válidos:\n\n" + "\n".join(f"- {e}" for e in errs))
    else:
        st.success("A es simétrica y definida positiva. Los datos son válidos.")


def p_escenarios():
    hdr("Tres escenarios", "Carga rápida y tabla comparativa obligatoria")
    cols = st.columns(3)
    for col, n in zip(cols, SC):
        if col.button(f"Cargar «{n}»", width="stretch", type="primary" if S.scen == n else "secondary"):
            load_scenario(n)
            st.rerun()
    cols = st.columns(3)
    for col, (n, sc) in zip(cols, SC.items()):
        col.markdown(f"<div class='card'><b>{n}</b><br>{sc['why']}</div>", unsafe_allow_html=True)
    R = scen_results()
    st.markdown("&nbsp;")
    st.markdown(f"#### Tabla comparativa (x₀ = 0, tol = {S.tol:.0e}, precondicionador {S.kind})")
    rows = {
        "Solución numérica de referencia": [vec(r["xref"]) for r in R.values()],
        "Solución GCP": [vec(r["gcp"]["x"]) for r in R.values()],
        "Error absoluto (Ea), por variable": [vec(np.abs(r["xref"] - r["gcp"]["x"]), "{:.1e}") for r in R.values()],
        "Error porcentual (Ep), por variable": [vec(100 * np.abs(r["xref"] - r["gcp"]["x"]) / np.abs(r["xref"]), "{:.1e}") for r in R.values()],
        "Iteraciones GC (sin precondicionar)": [r["gc"]["k"] for r in R.values()],
        "Iteraciones GCP (precondicionado)": [r["gcp"]["k"] for r in R.values()],
        "Número de condición κ(A)": [f"{r['kA']:.4g}" for r in R.values()],
        "Número de condición κ(M⁻¹A)": [f"{r['kM']:.4g}" for r in R.values()],
        "Residuo relativo final GCP": [f"{r['res_gcp']:.2e}" for r in R.values()],
        "¿Convergió el algoritmo?": [("Sí" if r["gcp"]["ok"] else "No") for r in R.values()],
        "Interpretación financiera": [interp(n, r) for n, r in R.items()],
    }
    df = pd.DataFrame(rows, index=list(R)).T.astype(str)
    st.table(df)
    st.download_button("Descargar tabla (CSV)", df.to_csv().encode("utf-8"), "tabla_comparativa.csv", "text/csv")
    fig = go.Figure()
    for (n, r), colr in zip(R.items(), ["#0d9488", "#7c3aed", "#f59e0b"]):
        fig.add_scatter(y=r["gc"]["hist"], name=f"GC · {n}", line=dict(color=colr, dash="dot"))
        fig.add_scatter(y=r["gcp"]["hist"], name=f"GCP · {n}", line=dict(color=colr, width=3))
    fig.update_yaxes(type="log", title="‖r‖₂/‖b‖₂")
    fig.update_xaxes(title="iteración")
    st.plotly_chart(style(fig, 380), width="stretch")


def interp(n, r):
    x, b = r["xref"], r["b"]
    if n == "Ideal":
        return (f"El gasto crece de {x[0]:.1f} (ene) a {x[-1]:.1f} (dic); el pico deseado de {b[-1]:.0f} se recorta a {x[-1]:.1f} y se traslada a los meses vecinos. "
                f"El sistema está bien condicionado (κ = {r['kA']:.2f}).")
    if n == "Bajo estrés":
        return (f"Con aguinaldo y fiestas, el deseado de diciembre sube a {b[-1]:.0f} y el plan asigna {x[-1]:.1f}; octubre y noviembre suben para absorberlo. "
                f"κ casi no cambia ({r['kA']:.2f}): el estrés desplaza el equilibrio sin dañar el sistema.")
    return (f"El plan queda casi plano ({x.min():.1f} a {x.max():.1f}), equivalente a una política de gasto constante. κ(A) = {r['kA']:.0f}: con meses fuertemente acoplados, "
            f"GC necesita {r['gc']['k']} iteraciones y GCP con {r['kind']} necesita {r['gcp']['k']}.")


def p_referencia():
    hdr("Solución de referencia", "Método directo: factorización de Cholesky A = LLᵀ, implementada desde cero")
    r, errs = current()
    if errs:
        return blocked(errs)
    L, x = r["L"], r["xref"]
    c = st.columns(3)
    kpi(c[0], "‖A − LLᵀ‖∞", f"{np.abs(r['A'] - L @ L.T).max():.1e}")
    kpi(c[1], "‖b − Ax_ref‖₂ / ‖b‖₂", f"{np.linalg.norm(r['b'] - r['A'] @ x) / np.linalg.norm(r['b']):.1e}")
    kpi(c[2], "Σx_ref vs Σb", f"{x.sum():.2f} vs {r['b'].sum():.2f}")
    st.markdown("&nbsp;")
    fig = go.Figure()
    fig.add_bar(x=MESES, y=r["b"], name="Presupuesto deseado p", marker_color="rgba(124,58,237,.45)")
    fig.add_scatter(x=MESES, y=x, name="Plan suavizado x_ref", line=dict(color="#0d9488", width=4), mode="lines+markers")
    st.plotly_chart(style(fig), width="stretch")
    a, b = st.columns([2, 3])
    a.markdown("**Vector x_ref**")
    a.dataframe(pd.DataFrame({"Variable": [f"x{i + 1} · {MESES[i]}" for i in range(len(x))], "x_ref": x}), hide_index=True, height=300, column_config={"x_ref": st.column_config.NumberColumn(format="%.6f")})
    b.markdown("**Factor L** (triangular inferior)")
    b.dataframe(pd.DataFrame(L, columns=[f"c{j + 1}" for j in range(len(L))]).round(4), height=300)

    st.markdown("#### Lectura de negocio: reglas y flujo de caja")
    g = st.columns(3)
    techo = g[0].number_input("Techo anual (miles Bs)", value=float(np.round(r["b"].sum(), 2)))
    piso = g[1].number_input("Mínimo mensual (miles Bs)", value=70.0)
    salto = g[2].number_input("Salto máximo entre meses (miles Bs)", value=40.0)
    dx = np.abs(np.diff(x))
    checks = [("Σx ≤ techo anual", x.sum() <= techo + 1e-6, f"Σx = {x.sum():.2f}"),
              ("x_i ≥ mínimo mensual", x.min() >= piso, f"mín = {x.min():.2f}"),
              ("|x_{i+1} − x_i| ≤ salto máximo", dx.max() <= salto, f"máx = {dx.max():.2f}")]
    for t, ok, d in checks:
        st.markdown(f"<span class='{'ok' if ok else 'bad'}'>{'✓ cumple' if ok else '✗ viola'}</span> &nbsp;{t} <small>({d})</small>", unsafe_allow_html=True)
    f2 = go.Figure()
    f2.add_scatter(x=MESES, y=np.cumsum(r["b"]), name="Acumulado deseado", line=dict(color="#7c3aed", dash="dot"))
    f2.add_scatter(x=MESES, y=np.cumsum(x), name="Acumulado del plan", line=dict(color="#0d9488", width=3), fill="tonexty", fillcolor="rgba(13,148,136,.12)")
    f2.update_yaxes(title="miles de Bs")
    st.plotly_chart(style(f2, 300), width="stretch")


ALG2 = """Datos: x0, r0 = b - A x0, M (precondicionador)
k = 0
mientras ||r_k||2 / ||b||2 >= tol  y  k < N_max:
    resolver  M z_k = r_k
    k = k + 1
    si k == 1:  p_1 = z_0
    sino:       beta_k = (r_{k-1}' z_{k-1}) / (r_{k-2}' z_{k-2});  p_k = z_{k-1} + beta_k p_{k-1}
    alpha_k = (r_{k-1}' z_{k-1}) / (p_k' A p_k)
    x_k = x_{k-1} + alpha_k p_k
    r_k = r_{k-1} - alpha_k A p_k
retornar x_k"""


def p_gcp():
    hdr("GCP vs GC", "Algoritmo 2 de Suñagua (2020) contra Gradientes Conjugados simple, solo como contraste")
    r, errs = current()
    if errs:
        return blocked(errs)
    if r["note"]:
        st.warning(r["note"])
    c = st.columns(4)
    kpi(c[0], "Iteraciones GC (sin M)", r["gc"]["k"])
    kpi(c[1], f"Iteraciones GCP ({r['kind']})", r["gcp"]["k"])
    kpi(c[2], "Residuo relativo GCP", f"{r['res_gcp']:.1e}")
    kpi(c[3], "¿Convergió GCP?", "Sí" if r["gcp"]["ok"] else "No")
    st.markdown("&nbsp;")
    fig = go.Figure()
    fig.add_scatter(y=r["gc"]["hist"], name="GC (sin precondicionar)", line=dict(color="#7c3aed", dash="dot", width=3))
    fig.add_scatter(y=r["gcp"]["hist"], name=f"GCP · {r['kind']}", line=dict(color="#0d9488", width=4))
    fig.add_hline(y=r["tol"], line_dash="dash", line_color="#e11d48", annotation_text="tolerancia")
    fig.update_yaxes(type="log", title="‖r_k‖₂ / ‖b‖₂")
    fig.update_xaxes(title="iteración")
    st.plotly_chart(style(fig), width="stretch")

    st.markdown("#### Comparación de precondicionadores sobre la matriz actual")
    rows = []
    for k in ["Ninguno (GC)", "Jacobi", "IC(0)", "SSOR"]:
        try:
            solve, M = make_precond(S.A, "Ninguno" if k.startswith("Ninguno") else k, S.omega)
            g = pcg(S.A, S.b, x0_used(), solve, S.tol, S.nmax)
            rows.append({"Precondicionador": k, "κ(M⁻¹A)": f"{cond_prec(S.A, M):.4g}", "Iteraciones": g["k"],
                         "Residuo final": f"{g['hist'][-1]:.2e}", "Cota teórica (ec. 1)": bound_iters(cond_prec(S.A, M), S.tol)})
        except ValueError as e:
            rows.append({"Precondicionador": k, "κ(M⁻¹A)": "—", "Iteraciones": "—", "Residuo final": str(e), "Cota teórica (ec. 1)": "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("La cota teórica acota el error en norma A, mientras que el criterio de parada usa el residuo. Sirve como referencia de orden de magnitud; además, GC termina a lo sumo en n pasos.")
    st.markdown("#### Vectores aproximados")
    n = len(r["xref"])
    st.dataframe(pd.DataFrame({"Variable": [f"x{i + 1}" for i in range(n)], "x_ref (Cholesky)": r["xref"], "x_GC": r["gc"]["x"], "x_GCP": r["gcp"]["x"]}),
                 hide_index=True, width="stretch", height=300, column_config={k: st.column_config.NumberColumn(format="%.6f") for k in ["x_ref (Cholesky)", "x_GC", "x_GCP"]})
    with st.expander("Algoritmo 2 (Suñagua, 2020) tal como está implementado"):
        st.code(ALG2, language="text")


def p_errores():
    hdr("Análisis de errores", "Error absoluto, relativo y porcentual por variable, residuo y condicionamiento")
    r, errs = current()
    if errs:
        return blocked(errs)
    df = error_table(r["xref"], r["gcp"]["x"])
    c = st.columns(4)
    kpi(c[0], "Residuo relativo ‖b−Ax‖₂/‖b‖₂", f"{r['res_gcp']:.2e}")
    kpi(c[1], "κ(A)", f"{r['kA']:.4g}")
    kpi(c[2], f"κ(M⁻¹A) · {r['kind']}", f"{r['kM']:.4g}")
    kpi(c[3], "Ep máximo (%)", f"{np.nanmax(df['Ep (%)']):.2e}")
    st.markdown("&nbsp;")
    st.dataframe(df, hide_index=True, width="stretch", height=470,
                 column_config={"x_ref": st.column_config.NumberColumn(format="%.6f"), "x_GCP": st.column_config.NumberColumn(format="%.6f"),
                                "Ea": st.column_config.NumberColumn(format="%.3e"), "Er": st.column_config.NumberColumn(format="%.3e"),
                                "Ep (%)": st.column_config.NumberColumn(format="%.3e")})
    if df["Er"].isna().any():
        st.info("El error relativo no está definido en las variables con x_ref = 0.")
    fig = go.Figure(go.Bar(x=df["Variable"], y=df["Ea"].clip(lower=1e-18), marker_color="#7c3aed"))
    fig.update_yaxes(type="log", title="Error absoluto Ea")
    st.plotly_chart(style(fig, 300), width="stretch")


def run_qa():
    rows = []

    def add(test, entrada, esperado, obtenido, ok):
        rows.append({"Prueba": test, "Entrada": entrada, "Resultado esperado": esperado, "Resultado obtenido": obtenido,
                     "Conclusión": "✓ Aprobada" if ok else "✗ Falló"})

    P = dict(tol=1e-6, nmax=200, kind="IC(0)")
    R = {n: run_all(build_A(sc["lam"]), sc["b"], np.zeros(N), P["tol"], P["nmax"], P["kind"]) for n, sc in SC.items()}
    for n, r in R.items():
        rel = np.linalg.norm(r["gcp"]["x"] - r["xref"]) / np.linalg.norm(r["xref"])
        add("1. Carga de escenarios", f"Escenario «{n}»", "GCP converge; error relativo vs Cholesky < 1e-4",
            f"convergió={r['gcp']['ok']}, it={r['gcp']['k']}, err_rel={rel:.1e}", r["gcp"]["ok"] and rel < 1e-4)

    A0, b0 = build_A(0.5), P_BASE.copy()

    def asimetrica(A, b):
        A[0, 1] += 5
        return A, b

    def no_definida(A, b):
        A[2, 2] = -1.0
        return A, b

    def b_nulo(A, b):
        return A, b * 0

    def celda_vacia(A, b):
        A[1, 1] = np.nan
        return A, b

    casos = [("A con a12 ≠ a21 (suma 5 a a12)", asimetrica, "simétrica"),
             ("A con a33 = −1 (no definida positiva)", no_definida, "definida positiva"),
             ("b = 0 (vector nulo)", b_nulo, "‖b‖₂ = 0"),
             ("Celda vacía (NaN) en A", celda_vacia, "vacías")]
    for desc, mod, clave in casos:
        try:
            errs = validate(*mod(A0.copy(), b0.copy()))
            crash = False
        except Exception as e:  # noqa: BLE001
            errs, crash = [str(e)], True
        add("2. Matriz no válida", desc, f"Error capturado que menciona «{clave}», sin colapsar", ("COLAPSÓ: " if crash else "") + (errs[0] if errs else "sin error"),
            (not crash) and any(clave in e for e in errs))

    r0 = R["Ideal"]
    A1, b1 = A0.copy(), b0.copy()
    A1[3, 3] += 1
    b1[5] += 10
    r1 = run_all(A1, b1, np.zeros(N), P["tol"], P["nmax"], P["kind"])
    cambio = np.linalg.norm(r1["xref"] - r0["xref"])
    rel = np.linalg.norm(r1["gcp"]["x"] - r1["xref"]) / np.linalg.norm(r1["xref"])
    add("3. Edición dinámica", "a44 += 1 y b6 += 10 sobre el escenario Ideal", "x_ref cambia; GCP coincide con la nueva x_ref (< 1e-4)",
        f"‖Δx_ref‖ = {cambio:.2f}, err_rel GCP = {rel:.1e}", cambio > 1e-6 and rel < 1e-4)

    rm = R["Mal condicionado"]
    add("4. Comparativa de precondicionamiento", "Escenario mal condicionado, IC(0), tol 1e-6", "it_GCP ≤ it_GC",
        f"it_GCP = {rm['gcp']['k']}, it_GC = {rm['gc']['k']}", rm["gcp"]["k"] <= rm["gc"]["k"])

    for n, r in R.items():
        d = abs(r["xref"].sum() - r["b"].sum())
        add("5. Invariante del modelo (extra)", f"Escenario «{n}»", "Σx = Σb (se conserva el presupuesto)", f"|Σx − Σb| = {d:.1e}", d < 1e-8 * r["b"].sum())

    Aa = np.array([[1e20, 1.0], [1.0, 1e-10]])
    k = cond_prec(Aa, np.diag(np.diag(Aa)))
    add("6. Ejemplo del artículo (extra)", "A = [[1e20, 1], [1, 1e-10]], M = diag(A)", "κ(M⁻¹A) = 1.000020000200002 (Suñagua, 2020)",
        f"κ(M⁻¹A) = {k:.15f}", abs(k - 1.000020000200002) < 1e-9)
    return pd.DataFrame(rows)


def p_qa():
    hdr("Módulo de pruebas (QA)", "Casos de prueba ejecutados en vivo con parámetros fijos: tol 1e-6, IC(0), x₀ = 0")
    df = run_qa()
    ok = (df["Conclusión"].str.startswith("✓")).sum()
    c = st.columns(3)
    kpi(c[0], "Pruebas ejecutadas", len(df))
    kpi(c[1], "Aprobadas", int(ok))
    kpi(c[2], "Fallidas", int(len(df) - ok))
    st.markdown("&nbsp;")
    st.dataframe(df, hide_index=True, width="stretch", height=min(80 + 38 * len(df), 700))
    st.caption("Las pruebas 1 a 4 son las obligatorias del enunciado. Las pruebas 5 y 6 son adicionales. La reactividad de la interfaz se comprueba en «Entrada de datos», donde cualquier cambio recalcula las demás secciones.")


def p_conclusiones():
    hdr("Conclusiones y referencias", "Interpretación técnica y financiera")
    R = scen_results()
    i, e, m = R["Ideal"], R["Bajo estrés"], R["Mal condicionado"]
    st.markdown(
        f"""
**Técnica.**
- Con λ = 0.5 el sistema está bien condicionado (κ(A) = {i['kA']:.2f}) y GC converge en {i['gc']['k']} pasos. En ese régimen precondicionar aporta poco.
- El estrés financiero modifica b pero no A: κ pasa de {i['kA']:.2f} a {e['kA']:.2f}. Solo se desplaza el equilibrio (diciembre: {i['xref'][-1]:.1f} → {e['xref'][-1]:.1f}).
- Con λ = 200, κ(A) = {m['kA']:.0f}. El problema es de **acople fuera de la diagonal**: la diagonal es casi constante, así que Jacobi casi no ayuda, como anticipa el artículo, y es IC(0) quien lleva κ(M⁻¹A) a {m['kM']:.3g}.
- Con A tridiagonal, IC(0) coincide con Cholesky exacto, por eso GCP converge en {m['gcp']['k']} iteración(es) frente a {m['gc']['k']} de GC. Es un caso particular favorable; en matrices con más relleno la mejora es menor pero real.
- La cota teórica (ec. 1) es pesimista para n pequeño, porque GC termina en a lo sumo n pasos.

**Financiera.**
- Suavizar tiene un costo y un beneficio: el plan se aleja del deseado en los picos, pero el flujo de caja es más estable. λ es la perilla de política.
- El invariante Σx = Σp garantiza que el suavizado redistribuye sin crear presupuesto.
- En estrés, la regla de salto máximo entre meses puede violarse (ver «Solución de referencia»): el modelo permite detectar cuándo la política pide algo que el negocio no tolera.
- Con λ muy alto el plan se vuelve casi plano: una política demasiado rígida ignora la estacionalidad real.

**Referencias.**
- Suñagua S., P. (2020). *Método de Gradientes Conjugados Precondicionado*. Revista Boliviana de Matemática – UMSA, 04, 2–7.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.
- Datos: presupuesto estacional ilustrativo, construido para este trabajo (ver «Caso de estudio»).
"""
    )


# ──────────────────────────────────────── Navegación ────────────────────────────────────────

PAGES = [
    (p_inicio, "Inicio", ":material/home:", "inicio"),
    (p_caso, "Caso de estudio", ":material/menu_book:", "caso"),
    (p_formulacion, "Formulación", ":material/functions:", "formulacion"),
    (p_entrada, "Entrada de datos", ":material/tune:", "entrada"),
    (p_escenarios, "Tres escenarios", ":material/layers:", "escenarios"),
    (p_referencia, "Solución de referencia", ":material/calculate:", "referencia"),
    (p_gcp, "GCP vs GC", ":material/speed:", "gcp"),
    (p_errores, "Análisis de errores", ":material/analytics:", "errores"),
    (p_qa, "Módulo de pruebas (QA)", ":material/fact_check:", "qa"),
    (p_conclusiones, "Conclusiones y referencias", ":material/article:", "conclusiones"),
]


def main():
    init()
    st.markdown(CSS, unsafe_allow_html=True)
    pg = st.navigation([st.Page(f, title=t, icon=i, url_path=u, default=(u == "inicio")) for f, t, i, u in PAGES], position="sidebar")
    errs = validate(S.A, S.b)
    with st.sidebar:
        st.markdown(f"**{S.scen}**")
        st.markdown(f"<small>tol {S.tol:.0e} · {S.kind} · n = {len(S.b)}</small>", unsafe_allow_html=True)
        st.markdown("<span class='bad'>✗ datos no válidos</span>" if errs else "<span class='ok'>✓ A simétrica y definida positiva</span>", unsafe_allow_html=True)
    pg.run()


main()
