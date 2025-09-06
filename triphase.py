import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from biorbd import marker_index
from casadi import MX, vertcat, sign
import os
import rerun as rr
import pickle

from bioptim import (
    Node,
    ConstraintList,
    InterpolationType,
    ConstraintFcn,
    ConfigureProblem,
    ObjectiveList,
    DynamicsFunctions,
    OptimalControlProgram,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    Objective,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    BiorbdModel,
    NonLinearProgram,
    ControlType,
    PhaseDynamics,
    OnlineOptim,
    ContactType,
    DynamicsEvaluation,
    PenaltyController,
    SolutionMerge,
    Axis,
    MultinodeConstraintList,
    PhaseTransitionList,
    PhaseTransitionFcn,
    DynamicsOptionsList,
    DynamicsOptions,
    MultinodeConstraintFcn,
    TorqueDynamics,
    TorqueBiorbdModel,
    ObjectiveWeight,
)

import shutil

#Define the stiffness and damping of the bar as global variables
stiffness = 14160
damping = 91

class DynamicModel(TorqueBiorbdModel):
    def __init__(self, biorbd_model_path):
        super().__init__(biorbd_model_path)

    def dynamics(
            self,
            time: MX,
            states: MX,
            controls: MX,
            parameters: MX,
            algebraic_states: MX,
            numerical_timeseries: MX,
            nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        Modify the system dynamics to model the bar as a spring

        Parameters
        ----------
        time: MX
            The current time of the system
        states: MX
            The current states of the system
        controls: MX
            The current controls of the system
        parameters: MX
            The current parameters of the system
        algebraic_states: MX
            The current algebraic states of the system
        numerical_timeseries: MX
            The current numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase of the ocp

        Returns
        -------
        The state derivative
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.controls["tau"], controls)


        tau[0] = -stiffness * q[0] + damping * qdot[0]  # x
        tau[1] = -stiffness * q[1] + damping * qdot[1]  # z


        qddot = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q, slope_qdot) * nlp.dt - vertcat(qdot, qddot)* nlp.dt

        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=defects)


def prepare_ocp(
        biorbd_model_path: str,
        final_time: tuple,
        n_shooting: tuple,
        min_time: float,
        max_time: float,
        total_mass: float,
        init_sol: bool,
        final_state_bound: bool,
        coef_fig : int,
        weight_control: float,
        weight_time: float = 1,
        mode: str="",
        ode_solver: OdeSolverBase = OdeSolver.RK4(),
        use_sx: bool = True,
        n_threads: int = 1,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics: bool = True,
        control_type: ControlType = ControlType.CONSTANT,
        x_init: InitialGuessList = None,
        u_init: InitialGuessList = None
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: tuple
        The initial guess for the final time of each phase
    n_shooting: tuple
        The number of shooting points (one number per phase) to define in the direct multiple shooting program
     min_time: float
        The minimum time allowed for the final node
    max_time: float
        The maximum time allowed for the final node
    total_mass : float
        The mass of the athlete
    init_sol : bool
        If it computes an initial solution
    final_state_bound : bool
        If the final state is with bound (false means it's with constraints)
    coef_fig : int
        Weighting coefficient for objectives that implement FIG code specifications
    weight_control: float
        Weight for the control minimization objective
    weight_time: float
        Weight for the time minimization objective
    mode : str
        Specifies the orientation of the pelvic during the descent phase
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls
    x_init : InitialGuessList
        Initial values of the states
    u_init : InitialGuessList
        Initial values of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = (DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path))

    # Index of useful degrees of freedom
    idx_RyHands = 2
    idx_elbow = 3
    idx_shoulder = 4
    idx_back = 5
    idx_RxThighR = 7
    idx_RyThighR = 8
    idx_RxThighL = 11
    idx_RyThighL = 12
    idx_KneeR = 9
    idx_KneeL = 13
    idx_FootR = 10
    idx_FootL = 14

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=weight_control, phase=0)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight_time, min_bound=min_time, max_bound=max_time,phase=0)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=weight_control, phase=1)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight_time, min_bound=min_time, max_bound=max_time,phase=1)

    objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=weight_control, phase=2)
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight_time, min_bound=min_time, max_bound=max_time,phase=2)

    if init_sol == False :
        # to stabilize the movement
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=2)

        # FIG code specifications (knees, elbows and ankles flexion and thighs abduction)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_elbow, target=0, weight=5*coef_fig, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_KneeR, target=0, weight=5*coef_fig, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_KneeL, target=0, weight=5*coef_fig, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_FootR, target=0, weight=1*coef_fig, phase=0)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_FootL, target=0, weight=1*coef_fig, phase=0)
        leg_weight = np.hstack((3*np.ones(26), np.zeros(25)))
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=idx_RxThighR, node=Node.ALL,weight=ObjectiveWeight(leg_weight, interpolation=InterpolationType.EACH_FRAME))
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=idx_RxThighL, node=Node.ALL,weight=ObjectiveWeight(leg_weight, interpolation=InterpolationType.EACH_FRAME))

        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_elbow, target=0, weight=5*coef_fig, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_KneeR, target=0, weight=5*coef_fig, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_KneeL, target=0, weight=5*coef_fig, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_FootR, target=0, weight=1*coef_fig, phase=1)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_FootL, target=0, weight=1*coef_fig, phase=1)

        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_elbow, target=0, weight=5*coef_fig, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_KneeR, target=0, weight=5*coef_fig, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_KneeL, target=0, weight=5*coef_fig, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_FootR, target=0, weight=1*coef_fig, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_FootL, target=0, weight=1*coef_fig, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_RxThighR, target=0, weight=3*coef_fig, phase=2)
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx_RxThighL, target=0, weight=3*coef_fig, phase=2)

    # Dynamics
    dynamics = DynamicsOptionsList()
    dynamics.add(DynamicsOptions(
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=5)
    ))
    dynamics.add(DynamicsOptions(
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=5)
    ))
    dynamics.add(DynamicsOptions(
        expand_dynamics=expand_dynamics,
        phase_dynamics=phase_dynamics,
        ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=5)
    ))

    # Constraints
    constraint_list = ConstraintList()
    if init_sol == False:
        # avoid the lower bar
        constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="LowerBarMarker",
                            second_marker="MarkerR", min_bound=0.02, max_bound=np.inf, axes=Axis.X, phase=1)
        constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="LowerBarMarker",
                            second_marker="MarkerL", min_bound=0.02, max_bound=np.inf, axes=Axis.X, phase=1)

        # impose the orientation of the pelvic during the descent phase
        if mode == "anteversion":
            # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.ALL, index=idx_RyThighL, min_bound=0, max_bound=np.inf, phase=0)
            # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.ALL, index=idx_RyThighR, min_bound=0, max_bound=np.inf, phase=0)
            # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.ALL, index=idx_back, min_bound=0, max_bound=np.inf, phase=0)
            constraint_list.add(ConstraintFcn.TRACK_MARKERS, node=Node.ALL, reference_jcs=idx_back, marker_index=3, axes=Axis.X, min_bound=-np.inf, max_bound=0, phase=0)


        elif mode == "retroversion":
            # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.ALL, index=idx_RyThighL, min_bound=-np.inf, max_bound=0, phase=0)
            # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.ALL, index=idx_RyThighR, min_bound=-np.inf, max_bound=0, phase=0)
            # constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.ALL, index=idx_back, min_bound=-np.inf, max_bound=0, phase=0)
            constraint_list.add(ConstraintFcn.TRACK_MARKERS, node=Node.ALL, reference_jcs=idx_back, marker_index=3, axes=Axis.X, min_bound=0, max_bound= np.inf,phase=0)

    # end of the first phase when the feet reach the height of the lower bar
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="LowerBarMarker",
                        second_marker="MarkerR", axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="LowerBarMarker",
                        second_marker="MarkerL", axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)

    #  symmetry of the thighs
    constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", node=Node.ALL_SHOOTING, first_dof=idx_RxThighR,second_dof=idx_RxThighL, coef=-1, phase=0)
    constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", node=Node.ALL_SHOOTING, first_dof=idx_RyThighR,second_dof=idx_RyThighL, coef=1, phase=0)
    constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", node=Node.ALL_SHOOTING, first_dof=idx_RxThighR,second_dof=idx_RxThighL, coef=-1, phase=1)
    constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", node=Node.ALL_SHOOTING, first_dof=idx_RyThighR,second_dof=idx_RyThighL, coef=1, phase=1)
    constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", node=Node.ALL_SHOOTING, first_dof=idx_RxThighR,second_dof=idx_RxThighL, coef=-1, phase=2)
    constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE, key="q", node=Node.ALL_SHOOTING, first_dof=idx_RyThighR,second_dof=idx_RyThighL, coef=1, phase=2)


    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=0)
    x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=0)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=1)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=1)
    x_bounds.add("q", bounds=bio_model[1].bounds_from_ranges("q"), phase=2)
    x_bounds.add("qdot", bounds=bio_model[1].bounds_from_ranges("qdot"), phase=2)


    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["q"][idx_RyHands, 0] = -2*np.pi/45 # hands tilted by 8° at the start
    x_bounds[0]["q"][1, 0] = - total_mass*9.81 / stiffness # equilibrium position
    x_bounds[0]["qdot"][:, 0] = 0  # speeds start at 0
    x_bounds[1]["q"][idx_RyHands, -1] = -np.pi # end of second phase with hands under the upper bar

    if final_state_bound :
        x_bounds[2]["q"][:, -1] = 0
        x_bounds[2]["q"][idx_RyHands, -1] = -2 * np.pi   # ends with hands 360° rotated
        x_bounds[2]["qdot"][idx_RyHands, -1] = - np.pi  # ends with hands speed of pi rad/s
    else : 
        idx_final_state = np.arange(idx_RyHands+1, idx_FootL+1) # index to constraint to 0 in the final state (all except those of the hands)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.END, index=idx_RyHands, min_bound=-2 * np.pi, max_bound=-2 * np.pi, phase=2)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", node=Node.END, index=idx_final_state, min_bound=-0.1, max_bound=0.1, phase=2)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", node=Node.END, index=idx_RyHands, min_bound=- np.pi-0.1, max_bound=- np.pi+0.1, phase=2)
        
    

    # Define control path bounds
    tau_min, tau_max = (-200, 200)
    n_tau = bio_model[0].nb_tau
    u_bounds = BoundsList()
    u_min = [tau_min] * n_tau
    u_max = [tau_max] * n_tau
    if init_sol == False:
        u_min[idx_RyHands] = 0
        u_max[idx_RyHands] = 0
        u_min[idx_shoulder] = -3.11*total_mass
        u_max[idx_shoulder] = 2.15*total_mass
        u_min[idx_RyThighR] = -4.20*total_mass
        u_max[idx_RyThighR] = 9.36*total_mass
        u_min[idx_RyThighL] = -4.20*total_mass
        u_max[idx_RyThighL] = 9.36*total_mass
    u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=0)
    u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=1)
    u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=2)

    if x_init == None:
        x_init = InitialGuessList()
        x_init.add("q", [0] * bio_model[0].nb_q, phase=0)
        x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=0)
        x_init.add("q", [0] * bio_model[0].nb_q, phase=1)
        x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=1)
        x_init.add("q", [0] * bio_model[0].nb_q, phase=2)
        x_init.add("qdot", [0] * bio_model[0].nb_qdot, phase=2)

    if u_init == None:
        u_init = InitialGuessList()
        u_init.add("tau", [0] * bio_model[0].nb_tau, phase=0)
        u_init.add("tau", [0] * bio_model[0].nb_tau, phase=1)
        u_init.add("tau", [0] * bio_model[0].nb_tau, phase=2)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
        constraints=constraint_list,

    )


def main():
    import os
    import pickle

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples/results")
    n_shooting = (50, 50, 50)

    num = 48
    filename = f"/applied_examples/athlete_{num}_deleva.bioMod"
    print("model : ", filename)
    masses = pd.read_csv(CURRENT_DIR+ "/applied_examples/masses.csv")
    masse = masses["total_mass"][num-1]

    # # initial solution
    # ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1, 0.5, 1), n_shooting=n_shooting, min_time=0.01, max_time=2, coef_fig=1, total_mass=masse, init_sol=True, weight_control=1, weight_time=0.1,final_state_bound=False, n_threads=32, use_sx=False)
    #
    # # --- Live plots --- #
    # ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration
    #
    # # --- Print ocp structure --- #
    # ocp.print(to_console=False, to_graph=False)
    #
    # # --- Solve the ocp --- #
    # solver = Solver.IPOPT()
    # solver.set_linear_solver("ma57")
    # solver.set_maximum_iterations(20000)
    # print("start solving")
    # sol = ocp.solve(solver)
    # print("solving finished")
    #
    #
    # qs = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['q']
    # qdots = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['qdot']
    # for i in range(1, len(sol.decision_states(to_merge=[SolutionMerge.NODES]))):
    #     qs = np.hstack((qs, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['q']))
    #     qdots = np.hstack((qdots, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['qdot']))
    #
    # taus = sol.decision_controls(to_merge=[SolutionMerge.NODES])[0]['tau']
    #
    # for i in range(1, len(sol.decision_controls(to_merge=[SolutionMerge.NODES]))):
    #     taus = np.hstack((taus, sol.decision_controls(to_merge=[SolutionMerge.NODES])[i]['tau']))
    #
    # # # --- Save the solution --- #
    # save = True
    #
    # if save == True:
    #     # Save the output of the optimization
    #     with open(os.path.join(RESULTS_DIR, f"athlete{num}_base.pkl"), "wb") as file:
    #         data = {"q": qs,
    #                 "qdot": qdots,
    #                 "tau": taus,
    #                 }
    #         pickle.dump(data, file)
    #
    #     print("initial solution saved")


    ######################################################################################################################################################
    with open(os.path.join(RESULTS_DIR, f"athlete{num}_base.pkl"), "rb") as file:
        prev_sol_data = pickle.load(file)

    qs = prev_sol_data["q"]
    qdots = prev_sol_data["qdot"]
    taus = prev_sol_data["tau"]

    n0, n1, n2 = n_shooting[0] + 1, n_shooting[1] + 1, n_shooting[2] + 1
    x_init = InitialGuessList()
    x_init.add("q", qs[:, :n0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("q", qs[:, n0:n0+n1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("q", qs[:, n0+n1:], interpolation=InterpolationType.EACH_FRAME, phase=2)
    x_init.add("qdot", qdots[:, :n0], interpolation=InterpolationType.EACH_FRAME, phase=0)
    x_init.add("qdot", qdots[:, n0:n0+n1], interpolation=InterpolationType.EACH_FRAME, phase=1)
    x_init.add("qdot", qdots[:, n0+n1:], interpolation=InterpolationType.EACH_FRAME, phase=2)

    u_init = InitialGuessList()
    u_init.add("tau", taus[:, :n0 - 1], interpolation=InterpolationType.EACH_FRAME, phase=0)
    u_init.add("tau", taus[:, n0 - 1:n0+n1-2], interpolation=InterpolationType.EACH_FRAME, phase=1)
    u_init.add("tau", taus[:, n0+n1-2:], interpolation=InterpolationType.EACH_FRAME, phase=2)


    #mode = "retroversion"
    mode = "anteversion"
    #mode=""

    # solution complete
    ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1, 0.5, 1), n_shooting=n_shooting,
                      min_time=0.01, max_time=2, total_mass=masse, init_sol=False, weight_control=0.0001, weight_time=1,
                      coef_fig=1,final_state_bound=False, mode=mode, n_threads=32, use_sx=False)


    # --- Live plots --- #
    ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration

    # --- Print ocp structure --- #
    ocp.print(to_console=False, to_graph=False)

    # --- Solve the ocp --- #
    solver = Solver.IPOPT()
    solver.set_linear_solver("ma57")
    solver.set_maximum_iterations(20000)
    sol = ocp.solve(solver)

    qs = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['q']
    qdots = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['qdot']
    for i in range(1, len(sol.decision_states(to_merge=[SolutionMerge.NODES]))):
        qs = np.hstack((qs, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['q']))
        qdots = np.hstack((qdots, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['qdot']))

    taus = sol.decision_controls(to_merge=[SolutionMerge.NODES])[0]['tau']
    for i in range(1, len(sol.decision_controls(to_merge=[SolutionMerge.NODES]))):
        taus = np.hstack((taus, sol.decision_controls(to_merge=[SolutionMerge.NODES])[i]['tau']))

    time = sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]

    # # --- Save the solution --- #
    save = True

    if save == True:
        # Save the output of the optimization
        with open(os.path.join(RESULTS_DIR, f"athlete{num}_complet_{mode}.pkl"), "wb") as file:
            data = {"q": qs,
                    "qdot": qdots,
                    "tau": taus,
                    "time": time,
                    }
            pickle.dump(data, file)

        print("data of full solution saved")



    # --- Animate the solution --- #
    # viewer = "bioviz"
    viewer = "pyorerun"
    sol.animate(n_frames=0, viewer=viewer, show_now=True)

    # --- Show the results graph --- #
    sol.print_cost()
    sol.graphs(show_bounds=True, show_now=True)

    with open(os.path.join(RESULTS_DIR, f"Sol_athlete{num}_complet_{mode}.pkl"), "wb") as file:
        del sol.ocp
        pickle.dump(sol, file)
    print("object solution of full solution saved")




if __name__ == "__main__":
    main()


