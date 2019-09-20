# Cloth_simulation based on Mass Spring Model

 The following simulations were based on Explicit Euler's Method with 
 1) K,stiffness equal to 200 and damping equal to 20
 <img src=https://github.com/Mypathissional/Cloth_simulation/blob/master/animations/Explicit_Euler_K%3D200.0_D%3D20_stiffness%3D200%2Cinterval_length0.1.gif>
 2) K,stiffness equal to 200 and damping equal to 0
 <img src=https://github.com/Mypathissional/Cloth_simulation/blob/master/animations/Explicit_Euler_K%3D200.0_D%3D0_stiffness%3D200%2Cinterval_length0.1.gif width="600" height="300"> 
To make your own animation adjust main.py
if save_to_file=True, the resulting animation is made and saved in the root
else 3d animation plot is show can be useful if one wants to see the cloth from different perspectives

Choose the solver from ODE_solvers.py
1)ExplicitEulerStep()
2)ImplicitEulerStep()
3)RK4Step()
4)AdamStep(f_prev, order) 
Then call solve__ODE
else if AdaptiveRungeKuttaFehlbergStep is used call solve_adaptive_ODE

