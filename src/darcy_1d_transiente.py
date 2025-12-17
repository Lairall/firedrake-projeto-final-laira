"""
Trabalho Final — Disciplina: GA 033 - Elementos Finitos
Tema: Escoamento monofásico compressível de gás ideal em meio poroso (1D)
Formulação variacional com Firedrake

Aluno(a): Laira Lopes Silva
Professor: Diego Volpatto
Instituição: LNCC

Descrição do problema:
----------------------
Resolve-se o problema transiente de escoamento monofásico de um gás ideal
em um meio poroso unidimensional, representando um reservatório de comprimento L.

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante considerada (Equação 5 do enunciado do projeto) é:

    φ ∂p/∂t = (k/μ) ∂/∂x ( p ∂p/∂x )

onde:
    p   = pressão
    φ   = porosidade
    k   = permeabilidade
    μ   = viscosidade do fluido

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)
    p = p_r  na fronteira do reservatório (x = L)

Condição inicial:
    p(x, 0) = p_r,  ∀ x ∈ Ω

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG),
e a discretização temporal é feita via esquema implícito de Euler.
"""

# ============================================================
# Importing libraries
# ============================================================

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Mesh definition
# ============================================================

# One-dimensional domain [0, L]
L = 50.0                 # domain length
numel = 200              # number of elements
x_left, x_right = 0.0, L

mesh = IntervalMesh(numel, x_left, x_right)

# ============================================================
# Function space declaration
# ============================================================

degree = 1               # Polynomial degree of approximation (Lagrange polynomial)
V = FunctionSpace(mesh, "CG", degree) # V_h space where pressure will be solved

# Auxiliary space for post-processing (projections)
Vref = FunctionSpace(mesh, "CG", 1)

# ============================================================
# Boundary conditions
# ============================================================

# Prescribed pressures (Dirichlet BCs)

# Injector well pressure at x = 0
p_w = Constant(2.0e7)  # 200 bar = 2e7 Pa
bc_left = DirichletBC(V, p_w, 1)

# Reservoir pressure at x = L
p_r = Constant(1.0e7)  # 100 bar = 1e7 Pa
bc_right = DirichletBC(V, p_r, 2)

# List of boundary conditions
bcs = [bc_left, bc_right]

# Unknown functions
p = Function(V, name="Pressure")        # pressure at time n+1
p_k = Function(V, name="Pressure_old")  # pressure at time n
v = TestFunction(V)

# ============================================================
# Initial condition
# ============================================================

p_k.assign(p_r)   # initial pressure in the reservoir
p.assign(p_r)


# ============================================================
# Physical parameters of the model
# ============================================================

phi = Constant(0.15)        # porosity
kappa = Constant(2.6647e-13)     # permeability [m^2]
mu = Constant(0.94e-5)         # viscosity [Pa.s]  in a temperature of 50C

# Source term (no sources)
f = Constant(0.0)

# ============================================================
# Time discretization parameters
# ============================================================

T_total = 4.0e7        # total simulation time [s] (~460 days)
num_steps = 400        # number of time steps
dt = Constant(T_total / num_steps)

# ============================================================
# Variational formulation (nonlinear residual)
# ============================================================

F = (
    phi * (p - p_k) / dt * v * dx
    + (kappa / mu) * p * dot(grad(p), grad(v)) * dx
    - f * v * dx     # f = 0
)

# ============================================================
# Nonlinear solver parameters (SNES)
# ============================================================

solver_parameters = {
    "snes_type": "newtonls",   # Método de Newton com line search
    "snes_rtol": 1e-8,          # Precisão relativa do resíduo
    "snes_atol": 1e-10,         # Precisão absoluta do resíduo
    "ksp_type": "preonly",      # Usar apenas o pré-condicionador
    "pc_type": "lu"             # Fatoração LU direta
}

# ============================================================
# Time loop
# ============================================================

t = 0.0
for n in range(num_steps):
    t += float(dt)

    solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)
    
    # Numerical checks
    p_array = p.dat.data_ro
    p_mid = p.at(L/2)

    print(f"Time step {n+1}/{num_steps}, time = {t/86400:.2f} days")
    print(f"  p_min = {p_array.min():.3e} Pa, p_max = {p_array.max():.3e} Pa")
    print(f"  p(x=L/2) = {p_mid:.3e} Pa")
    
    # p_array = p.dat.data_ro
    # print(f"Time step {n+1}/{num_steps}, time = {t/86400:.2f} days")
    # print(f"  p_min = {p_array.min():.3e} Pa, p_max = {p_array.max():.3e} Pa")

    # Update solution for next time step
    p_k.assign(p)

# ============================================================


