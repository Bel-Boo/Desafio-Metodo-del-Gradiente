# Gradientes Conjugados Precondicionado en finanzas

**Suavizador de presupuesto anual** · Ingeniería de Sistemas · Métodos Numéricos
Universidad Mayor de San Andrés · Carrera de Informática

Aplicación web (Streamlit) que modela la distribución del gasto de 12 meses como un sistema lineal
simétrico definido positivo **Ax = b** y lo resuelve con:

- **Cholesky propio** como solución numérica de referencia (método directo).
- **Gradientes Conjugados (GC)** simple, solo como contraste.
- **Gradientes Conjugados Precondicionado (GCP)**, siguiendo el Algoritmo 2 de Suñagua (2020), con precondicionadores Jacobi, IC(0) y SSOR.

Incluye los tres escenarios obligatorios (Ideal, Bajo estrés, Mal condicionado), análisis de errores y condicionamiento, y un módulo de pruebas (QA) que se ejecuta en vivo.

> Datos del estudiante: edita las constantes `ESTUDIANTE`, `CODIGO`, `MENCION`, `ASIGNATURA` y `UNIVERSIDAD` al inicio de `app.py`.

---

## El modelo

Se minimiza el costo del plan de gasto `x` (miles de Bs, 12 meses):

```
f(x) = ½‖x − p‖² + (λ/2) Σ (x[i+1] − x[i])²
```

- `p`: presupuesto deseado (estacional, con pico en diciembre).
- `λ`: rigidez de la política de suavizado.
- La condición de primer orden da **(I + λT)x = p**, con `T` tridiagonal `[−1, 2, −1]` (extremos en 1).

**Por qué A es SPD:** `xᵀAx = ‖x‖² + λΣ(x[i+1] − x[i])² > 0` para todo `x ≠ 0`.
**Invariante:** como `1ᵀT = 0`, se cumple `Σx = Σp`: el suavizado redistribuye el presupuesto, no lo crea.

| Escenario | Cambio | κ(A) |
|---|---|---|
| Ideal | λ = 0.5 | 2.97 |
| Bajo estrés | λ = 0.5, b de noviembre y diciembre ×1.4 | 2.97 |
| Mal condicionado | λ = 200 (acople fuerte fuera de la diagonal) | 787.4 |

En el escenario mal condicionado el problema es de acople y no de escala, por eso se usa **IC(0)** (Jacobi casi no mejora porque la diagonal es casi constante). Con A tridiagonal, IC(0) coincide con Cholesky exacto y GCP converge en 1 iteración; la app permite comparar los tres precondicionadores.

---

## Estructura del repositorio

```
.
├── app.py                  # Aplicación completa (núcleo numérico + interfaz)
├── requirements.txt        # Dependencias
├── README.md
└── .streamlit/
    └── config.toml         # Tema (color primario, fuente monoespaciada)
```

### Arquitectura de `app.py`

1. **Modelo financiero:** `build_A`, escenarios `SC`, datos `P_BASE`.
2. **Núcleo numérico (sin librerías de álgebra para el método):** `cholesky`, `fsub`/`bsub`, `ic0`, `make_precond`, `pcg` (Algoritmo 2), `cond_prec`, `bound_iters`, `validate`, `run_all`.
3. **Estado global** en `st.session_state` (A, b, x₀, tolerancia, precondicionador).
4. **Páginas** (una función por sección) y navegación con `st.navigation`.

`numpy` solo se usa para productos matriciales y autovalores (κ).

### Secciones de la web

Inicio · Caso de estudio · Formulación · Entrada de datos · Tres escenarios · Solución de referencia · GCP vs GC · Análisis de errores · Módulo de pruebas (QA) · Conclusiones y referencias.

---

## Ejecución local

Requisitos: Python 3.10 o superior.

```bash
git clone <URL-de-tu-repositorio>
cd <carpeta-del-repositorio>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Se abre en `http://localhost:8501`. El tema claro u oscuro sigue la configuración de tu navegador o del menú de Streamlit (⋮ → Settings).

## Publicación (Streamlit Community Cloud)

1. Sube el repositorio a GitHub (público).
2. Entra a <https://share.streamlit.io> e inicia sesión con GitHub.
3. **New app** → elige el repositorio, la rama `main` y el archivo principal `app.py`.
4. **Deploy**. El enlace público resultante es el que se entrega.

---

## Pruebas (módulo QA)

La sección «Módulo de pruebas (QA)» ejecuta con parámetros fijos (tol 10⁻⁶, IC(0), x₀ = 0):

| # | Prueba | Resultado esperado |
|---|---|---|
| 1 | Carga de escenarios | GCP converge y coincide con Cholesky en los tres |
| 2 | Matriz no válida (asimétrica, no definida positiva, b = 0, celda vacía) | Error capturado y notificado sin colapsar |
| 3 | Edición dinámica | Cambiar a₄₄ y b₆ recalcula x_ref y GCP |
| 4 | Comparativa de precondicionamiento | it_GCP ≤ it_GC en el escenario mal condicionado |
| 5 | Invariante Σx = Σb (extra) | Se conserva en los tres escenarios |
| 6 | Ejemplo 2×2 del artículo (extra) | κ(M⁻¹A) = 1.000020000200002 |

## Uso rápido

1. **Tres escenarios:** carga Ideal, Bajo estrés o Mal condicionado y mira la tabla comparativa.
2. **Entrada de datos:** edita celdas de A y b (con simetría automática activada, al editar aᵢⱼ se actualiza aⱼᵢ), cambia tolerancia, x₀ o precondicionador. Las demás secciones se recalculan solas.
3. Prueba romper la matriz (por ejemplo, un valor distinto en a₁₂ y a₂₁ sin simetría automática): la web lo detecta y explica el error.

## Sugerencia de commits

```
feat: estructura inicial del proyecto
feat: modelo financiero y matrices de los tres escenarios
feat: Cholesky y validación SPD
feat: GC y GCP (Algoritmo 2) con Jacobi, IC(0) y SSOR
feat: análisis de errores y números de condición
feat: interfaz, navegación y estilos
feat: módulo QA
docs: README
```

## Referencias

- Suñagua S., P. (2020). *Método de Gradientes Conjugados Precondicionado*. Revista Boliviana de Matemática – UMSA, 04, 2–7.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press.
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM.

Los datos de presupuesto son ilustrativos (estacionalidad típica de una empresa), construidos para este trabajo.
