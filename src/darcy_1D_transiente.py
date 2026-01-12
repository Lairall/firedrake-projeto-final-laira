"""
Trabalho Final — Disciplina: GA 033 - Elementos Finitos
Tema: Escoamento monofásico compressível de gás ideal em meio poroso (1D)
Formulação variacional com Firedrake

Descrição do problema: (transiente)
----------------------
Resolve-se o problema transiente de escoamento monofásico de um gás ideal
em um meio poroso unidimensional, representando um reservatório de comprimento L.

Admite-se que:
- o meio é rígido (porosidade constante),
- o gás é ideal (fator de compressibilidade Z = 1),
- não há termo fonte,
- efeitos gravitacionais são desprezados.

A equação governante considerada é:

    φ ∂p/∂t = (k/μ) ∂/∂x ( p ∂p/∂x )

onde:
    p   = pressão
    φ   = porosidade
    k   = permeabilidade
    μ   = viscosidade do fluido

Condições de contorno:
    p = p_w  na fronteira do poço injetor (x = 0)

Condição inicial:
    p(x, 0) = p_r,  ∀ x ∈ Ω

A discretização espacial é realizada pelo Método dos Elementos Finitos
utilizando elementos de Lagrange contínuos (CG),
e a discretização temporal é feita via esquema implícito de Euler.
"""

# Importing libraries
from firedrake import *
import numpy as np
import matplotlib.pyplot as plt

# Mesh definition
numel = 200
L = 50.0
x_left, x_right = 0.0, L
mesh = IntervalMesh(numel, x_left, x_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
Vref = FunctionSpace(mesh, "CG", 1)

# Boundary condition (Dirichlet) and Initial condition
boundary_value_left = 2e7
bc_left = DirichletBC(V, boundary_value_left, 1)  # Boundary condition in 1 marked bounds (left)
bcs = [bc_left]

ic = Constant(1e7)


# Trial and Test functions
p = Function(V)
p_k = Function(V)
v = TestFunction(V)

# Physical parameters
phi = Constant(0.15)        # porosity
kappa = Constant(2.6647e-13)     # permeability [m^2]
mu = Constant(0.94e-5)         # viscosity [Pa.s]  in a temperature of 50C
f = Constant(0.0)            # source term  

# Time parameters
T_total = 4.147e7  # 480 days
dt = T_total / 500.

# Assigning the IC
p_k.assign(ic)
p.assign(ic)

# Compressibility factor fitted from PR-EoS in terms of pressure
def Z(p):
    return 1.0

# Non-linear pressure term
def fp(p):
    return p / Z(p)

# Residual variational formulation
F = phi * inner((fp(p) - fp(p_k)) / dt, v) * dx + (kappa / mu) * inner(fp(p) * grad(p), grad(v)) * dx
F -= f * v * dx

# Solver parameters
solver_parameters = {
    'mat_type': 'aij',
    'snes_tyoe': 'newtonls',
    'pc_type': 'lu'
}

# Iterating and solving over the time
t = dt
step = 0
# diego: x_values = mesh.coordinates.vector().dat.data
x_values = mesh.coordinates.dat.data_ro # Laira

sol_values = []
p_values_deg1 = []
psol_deg1 = Function(Vref)
while t <= T_total:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    solve(F == 0, p, bcs=bcs, solver_parameters=solver_parameters)
    # diego : sol_vec = np.array(p.vector().dat.data)
    # sol_values.append(sol_vec)
    # psol_deg1.project(p)
    # p_vec_deg1 = np.array(psol_deg1.vector().dat.data)
    # p_values_deg1.append(p_vec_deg1)

    sol_vec = p.dat.data_ro.copy()
    sol_values.append(sol_vec)

    psol_deg1.project(p)
    p_vec_deg1 = psol_deg1.dat.data_ro.copy()
    p_values_deg1.append(p_vec_deg1)
    p_k.assign(p)

    t += dt

# *** Plotting ***

# Setting up the figure object
fig = plt.figure(dpi=300, figsize=(8, 6))
ax = plt.subplot(111)

# Plotting the data
steps_to_plot = [1, 10, 30, 60, 120, 360, 480]
for i in steps_to_plot:
    ax.plot(x_values, p_values_deg1[i-1] / 1e3, label=('Day %i' % (i)))

# Getting and setting the legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Setting the xy-labels
plt.xlabel(r'$x$ [m]')
plt.ylabel(r'Pressure [kPa]')
plt.xlim(x_values.min(), x_values.max())

# Setting the grids in the figure
plt.minorticks_on()
plt.grid(True)
plt.grid(False, linestyle='--', linewidth=0.5, which='major')
plt.grid(False, linestyle='--', linewidth=0.1, which='minor')

# Displaying the plot
plt.tight_layout()
plt.savefig('compressible-flow-transiente.png')
#plt.show()