# VEM
Virtual Element Method for cut static background meshes, interface modelled by the level set method

Run the "study" files to see what is going on.

# Installation

Add VEM to your path by creating a file "/root/.julia/config/startup.jl" that includes the line

```push!(LOAD_PATH, "/root/your_path_to_VirtuE/src")```

In the Julia terminal run pwd() to see how your particular root path looks. 

# Coding principles
Let's try to only implement what we need. No need for placeholders, we can always copy things when we need them

No need to implement more than necessary ( lowest order is fine if you want lowest order )

Every time you write code twice, you should hear alarm bells ringing

Readability comes before performance

Each line should have just one operation and each function should have just one functionality

Let's try to only have running scripts when we push


<!-- # LSM:
(1.5 Write more stable time FD, optionally needed)
2. Fix Reinit
3. Get method for calculating curvature flow
4. Implement ways of getting curves.. need sampled data -->

# TODO:

Find general way to compute diameter from coordinates ✓

Make sparse instead of boolean, findall ✓

Implement rudimentary plotting ✓

Implement quadrature integration (using MiniQhull) ✓

Permeability constant depend on each element? Implement ✓

Setup cutfem with the same geometry and problem, compare condition numbers ✓

Implement proper L2 norm using L2 projection of VEM sol ✓

Implement natural macro elements ✓

Rename mesh.nodes to something like mesh.coords ✓

Mixed VEM ✓

Make code files into Modules for readability (where is that function from?) ✓

Erik: Assembly for "cutVEM"

Look inside Monomials.jl and rename so that some are proper functions, and some are eval_...

Find a benchmark mu field radial example (or Hansbo example)

Script that exports Julia->Paraview

Auxiliary space preconditioning

Erik: Read article auxiliary space preconditioning

Higher order k=2 VEM
