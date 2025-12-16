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
V = FunctionSpace(mesh, "CG", degree) # space where pressure will be solved

# Auxiliary space for post-processing (projections)
Vref = FunctionSpace(mesh, "CG", 1)

# ============================================================
# Boundary conditions
# ============================================================

# Prescribed pressure at the inlet (x = 0)
p_w = Constant(2.0e7)  # 200 bar = 2e7 Pa
bc_left = DirichletBC(V, p_w, 1)
bcs = [bc_left]

# Unknown functions
p = Function(V, name="Pressure")        # pressure at time n+1
p_k = Function(V, name="Pressure_old")  # pressure at time n
v = TestFunction(V)

# ============================================================
# Initial condition
# ============================================================

p_r = Constant(1.0e7)  # 100 bar = 1e7 Pa
p_k.assign(p_r)    
p.assign(p_r)           