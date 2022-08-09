using LinearAlgebra
using SparseArrays
using IterativeSolvers

using Revise
using VirtuE

## -------------- (u,v)+(divu,divv) refinement tests --------------#
k = 0 # Polynomial degree
N = 10 # size of mesh

eps = 5 * 10.0^-2

mesh = Meshing.create_tri_mesh(N)

levelset(x) = x[2] - (0.5 + eps)
mesh = Meshing.remesh(mesh, levelset)

M = AuxPrecond.assemble_mixed_energy_matrix(mesh, k)
P = AuxPrecond.AuxPreconditioner(mesh)

b = randn(size(M, 1))
restart = size(b, 1)

x_unpr, log_unpr = gmres(M, b, restart=restart, log=true)
x_prec, log_prec = gmres(M, b, Pl=P, restart=restart, log=true)

println("Unpreconditioned: ", log_unpr)
println("Preconditioned: ", log_prec)
