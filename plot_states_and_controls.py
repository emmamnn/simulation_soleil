import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


num = 576
mode = "retroversion"
#mode = "anteversion"
#mode=""

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples/results")
filename = f"athlete{num}_complet_{mode}.pkl"
with open(os.path.join(RESULTS_DIR, filename), "rb") as file:
    data = pickle.load(file)

q = data["q"]
qdot = data["qdot"]
tau = data["tau"]
t = data["time"]


nb_tau= np.shape(tau)[1]
t_tau = np.arange(0,nb_tau,1)

dof = ["Tx mains","Tz mains", "Ry mains", "Ry coudes", "Ry epaules", "Ry dos","Ry tête", "Rx thighR", "Ry thighR",
       "Ry genouR","Ry piedR","Rx thighL", "Ry thighL","Ry genouL", "Ry piedL"]

plt.figure(1)
for i in range (9):
    plt.subplot(3,3,i+1)
    #plt.plot(t, q1[i], label="Valeurs de base")
    plt.plot(t, q[i], label= "q")
    plt.xlabel("Temps (s)")
    plt.ylabel("Position (rad)")
    titre = dof[i]
    plt.title(f"q {titre}")
    plt.grid()
    plt.legend()

plt.figure(2)
for i in range (9,15):
    plt.subplot(3,3,i-8)
    #plt.plot(t, q1[i], label="Valeurs de base")
    plt.plot(t, q[i], label= "q")
    plt.xlabel("Temps (s)")
    plt.ylabel("Position (rad)")
    titre = dof[i]
    plt.title(f"q {titre}")
    plt.grid()
    plt.legend()



plt.figure(3)
for i in range (9):
    plt.subplot(3,3,i+1)
    #plt.step(ttau, tau1[i], where = "pre", label="Valeurs limitées")
    plt.step(t_tau, tau[i],where = "pre", label= "tau")
    plt.xlabel("Noeuds")
    plt.ylabel("Couple (Nm)")
    titre = dof[i]
    plt.title(f"tau {titre}")
    plt.grid()
    plt.legend()

plt.figure(4)
for i in range (9,15):
    plt.subplot(3,3,i-8)
    #plt.step(ttau, tau1[i], where = "pre", label="Valeurs limitées")
    plt.step(t_tau, tau[i],where = "pre", label= "tau")
    plt.xlabel("Noeuds")
    plt.ylabel("Couple (Nm)")
    titre = dof[i]
    plt.title(f"tau {titre}")
    plt.grid()
    plt.legend()

plt.figure(5)
for i in range (9):
    plt.subplot(3,3,i+1)
    #plt.plot(t, q1[i], label="Valeurs de base")
    plt.plot(t, qdot[i], label= "qdot")
    plt.xlabel("Temps")
    plt.ylabel("Vitesse (rad/s)")
    titre = dof[i]
    plt.title(f"q {titre}")
    plt.grid()
    plt.legend()

plt.figure(6)
for i in range (9,15):
    plt.subplot(3,3,i-8)
    #plt.plot(t, q1[i], label="Valeurs de base")
    plt.plot(t, qdot[i], label= "qdot")
    plt.xlabel("Temps")
    plt.ylabel("Vitesse (rad/s)")
    titre = dof[i]
    plt.title(f"qdot {titre}")
    plt.grid()
    plt.legend()

plt.show()


