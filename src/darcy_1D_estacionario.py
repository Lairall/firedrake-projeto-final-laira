"""
Trabalho Final — Disciplina: GA 033 - Elementos Finitos
Tema: Escoamento monofásico compressível de gás ideal em meio poroso (1D)
Formulação variacional com Firedrake

Descrição do problema: (estacionário)
----------------------
Resolve-se o problema estacionário de escoamento monofásico de um gás ideal
em um meio poroso unidimensional, representando um reservatório de comprimento L.

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante considerada (Equação 5 do enunciado do projeto) é:

    (k/μ) ∂/∂x ( p ∂p/∂x ) = 0

onde:
    p   = pressão
    φ   = porosidade
    k   = permeabilidade
    μ   = viscosidade do fluido

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)
    p = p_r  na fronteira do reservatório (x = L)

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG), 
"""

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh definition
numel = 200
L = 50.0
mesh = IntervalMesh(numel, 0.0, L)

# Function space
degree = 1
V = FunctionSpace(mesh, "CG", degree)

# Boundary conditions
p_left = Constant(2.0e7)   # 200 bar
p_right = Constant(1.0e7)  # 100 bar

bc_left = DirichletBC(V, p_left, 1)
bc_right = DirichletBC(V, p_right, 2)

bcs = [bc_left, bc_right]

# Unknown and test function
p = Function(V, name="Pressure")
v = TestFunction(V)

# Initial guess
p.assign(1.5e7)

# Physical parameters
kappa = Constant(1.0e-18)
mu = Constant(0.94e-5)

# Compressibility (gás ideal)
def Z(p):
    return 1.0

def fp(p):
    return p / Z(p)

# Variational formulation (STEADY STATE)
F = (kappa / mu) * inner(fp(p) * grad(p), grad(v)) * dx

# Solver parameters
solver_parameters = {
    "mat_type": "aij",
    "snes_type": "newtonls",
    "pc_type": "lu"
}

# Solve
solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)

# Plotting
x_values = mesh.coordinates.dat.data_ro
p_values = p.dat.data_ro / 1e3  # kPa

plt.figure(dpi=300, figsize=(8, 6))
plt.plot(x_values, p_values, label="Steady state")
plt.xlabel(r"$x$ [m]")
plt.ylabel("Pressure [kPa]")
plt.xlim(x_values.min(), x_values.max())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("compressible-flow-steady-2dirichlet.png")
# plt.show()
