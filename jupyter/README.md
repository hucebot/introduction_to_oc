## Simple LIPM

These exercises consist of developing the LIPM model (`exercises/lipm.ipynb`) and solving the linear optimal control problem (L-OCP) resulting from tracking 
a ZMP reference trajectory along the $Y$ axis (`zmp_ref.ipynb`).

In particular, the file `exercises/eQP.ipynb` solves the L-OCP by directly solving the KKT system of the resulting equality-constrained QP. 
In contrast, `exercises/riccati.ipynb` uses a Riccati factorization of the KKT system to solve the L-OCP in a stage-wise fashion.

