"""
free decay time

Ryosuke Nakashima
2020.12.21
"""

import datetime
import math
import numpy as np
import matplotlib.pyplot as plt

# Bessel function
from scipy.special import jv
from scipy.special import yv

# secant method
from scipy import optimize

# Tex
from matplotlib import rc
rc('text', usetex=True)

# start time
now = datetime.datetime.now()
print('start: ', now)

" ====== parameter ====== "
# 0: toroidal free decay mode
# 1: poloidal free decay mode
switch_tp = 1 # 0 or 1

# outer boundary condition
# 0: insulator
# 1: perfect conductor
switch_obc = 0 # 0 or 1

# inner boundary condition
# 0: insulator
# 1: perfect conductor
switch_ibc = 0 # 0 or 1

# radius ratio
eta_init = 0.01
eta_step = 0.003
eta_end = 0.99

# degree
n_init = 1
n_step = 1
n_end = 5
color_n = ["red", "magenta", "green", "cyan", "blue"]

# decay_time
lambda_sqrt_init = 0
lambda_sqrt_end = 30

# iteration number (secant method)
num_iter = 100

fig_name = "./free-decay_time_"
" === end [parameter] === "

num_eta = 1 + math.floor( (eta_end - eta_init)/eta_step )
lin_eta = np.linspace(eta_init, eta_end, num_eta)

num_n = 1 + math.floor( (n_end - n_init)/n_step )
lin_n = np.linspace(n_init, n_end, num_n)

fig1 = plt.figure( figsize=(6, 6) )
ax1 = fig1.add_subplot(111)

# toroidal free decay mode & poloidal free decay mode (Ipc/Opc, FS/Opc)
if (switch_tp == 0) or \
    ( (switch_tp == 1) and (switch_obc == 1) and (switch_ibc == 1) ):

    fig_name = fig_name+"t.png"
    fig_title = "Toroidal free decay mode / "+ \
        "Poloidal free decay mode (Ipc/Opc, FS/Opc)"

    for i_n in range(num_n):
        n = lin_n[i_n]

        # full sphere
        f_fs = lambda x_sqrt: jv(n+1/2, x_sqrt)
        f_fs_prime = lambda x_sqrt: \
            - jv(n+3/2, x_sqrt) + (n+1/2) * jv(n+1/2, x_sqrt) / x_sqrt

        norm_x_sqrt_fs = np.random.random(num_iter)
        x_sqrt_fs = (lambda_sqrt_end - lambda_sqrt_init) \
            *  norm_x_sqrt_fs + lambda_sqrt_init
        root_fs = optimize.newton(f_fs, x_sqrt_fs, fprime=f_fs_prime, \
            maxiter=1000)

        for i_iter in range(num_iter):
            if (root_fs[i_iter] > lambda_sqrt_init+0.1) \
                and (root_fs[i_iter] < lambda_sqrt_end):

                ax1.scatter(0, root_fs[i_iter], \
                    s=30, c=color_n[i_n], marker="*", clip_on=False)
            #
        #

        for i_iter in range(num_iter):

            # spherical shell
            f = lambda x_sqrt, eta: \
                jv(n+1/2, eta * x_sqrt) \
                * yv(n+1/2, x_sqrt) \
                - jv(n+1/2, x_sqrt) \
                * yv(n+1/2, eta * x_sqrt)

            norm_x_sqrt = np.random.random(len(lin_eta))
            x_sqrt = (lambda_sqrt_end - lambda_sqrt_init) \
                * norm_x_sqrt + lambda_sqrt_init

            root = optimize.newton(f, x_sqrt, args=(lin_eta, ), maxiter=1000)
            if i_iter == 0:
                ax1.scatter(lin_eta, root+2*lambda_sqrt_end, s=20, \
                    c=color_n[i_n], label="$n=$ "+str(i_n+1))
            else:
                ax1.scatter(lin_eta, root, s=1, c=color_n[i_n])
            #
        #
    #
#

# poloidal free decay mode
if switch_tp == 1:

    # Ii/Oi
    if (switch_obc == 0) and (switch_ibc == 0):
        fig_name = fig_name+"p_IiOi.png"
        fig_title = "Poloidal free decay mode (Ii/Oi)"
    # Ii/Opc
    elif (switch_obc == 1) and (switch_ibc == 0):
        fig_name = fig_name+"p_IiOpc.png"
        fig_title = "Poloidal free decay mode (Ii/Opc)"
    # Ipc/Oi
    elif (switch_obc == 0) and (switch_ibc == 1):
        fig_name = fig_name+"p_IpcOi.png"
        fig_title = "Poloidal free decay mode (Ipc/Oi)"
    #

    for i_n in range(num_n):
        n = lin_n[i_n]

        # full sphere
        # Ii/Opc
        if (switch_obc == 1) and (switch_ibc == 0):
            f_fs = lambda x_sqrt: jv(n+1/2, x_sqrt)
            f_fs_prime = lambda x_sqrt: \
                - jv(n+3/2, x_sqrt) + (n+1/2) * jv(n+1/2, x_sqrt) / x_sqrt
        # Ii/Oi, Ipc/Oi
        else:
            f_fs = lambda x_sqrt: jv(n-1/2, x_sqrt)
            f_fs_prime = lambda x_sqrt: \
                - jv(n+1/2, x_sqrt) + (n-1/2) * jv(n-1/2, x_sqrt) / x_sqrt
        #
        norm_x_sqrt_fs = np.random.random(num_iter)
        x_sqrt_fs = (lambda_sqrt_end - lambda_sqrt_init) \
            *  norm_x_sqrt_fs + lambda_sqrt_init
        root_fs = optimize.newton(f_fs, x_sqrt_fs, fprime=f_fs_prime, \
            tol=1e-10, maxiter=1000)

        for i_iter in range(num_iter):
            if (root_fs[i_iter] > lambda_sqrt_init+0.1) \
                and (root_fs[i_iter] < lambda_sqrt_end):

                ax1.scatter(0, root_fs[i_iter], \
                    s=30, c=color_n[i_n], marker="*", clip_on=False)
            #
        #

        for i_iter in range(num_iter):

            # spherical shell
            # Ii/Oi
            if (switch_obc == 0) and (switch_ibc == 0):
                f = lambda x_sqrt, eta: \
                    ( (2*n+1) * jv(n+1/2, eta * x_sqrt) / ( eta * x_sqrt ) \
                    - jv(n-1/2, eta * x_sqrt) ) \
                    * yv(n-1/2, x_sqrt) \
                    - ( (2*n+1) * yv(n+1/2, eta * x_sqrt) / ( eta * x_sqrt ) \
                    - yv(n-1/2, eta * x_sqrt) ) \
                    * jv(n-1/2, x_sqrt)
            # Ii/Opc
            elif (switch_obc == 1) and (switch_ibc == 0):
                f = lambda x_sqrt, eta: \
                    ( (2*n+1) * jv(n+1/2, eta * x_sqrt) / ( eta * x_sqrt ) \
                    - jv(n-1/2, eta * x_sqrt) ) \
                    * yv(n+1/2, x_sqrt) \
                    - ( (2*n+1) * yv(n+1/2, eta * x_sqrt) / ( eta * x_sqrt ) \
                    - yv(n-1/2, eta * x_sqrt) ) \
                    * jv(n+1/2, x_sqrt)
            # Oi/Ipc
            elif (switch_obc == 0) and (switch_ibc == 1):
                f = lambda x_sqrt, eta: \
                    jv(n+1/2, eta * x_sqrt) \
                    * yv(n-1/2, x_sqrt) \
                    - jv(n-1/2, x_sqrt) \
                    * yv(n+1/2, eta * x_sqrt)
            #

            norm_x_sqrt = np.random.random(len(lin_eta))
            x_sqrt = (lambda_sqrt_end - lambda_sqrt_init) \
                * norm_x_sqrt + lambda_sqrt_init

            root = optimize.newton(f, x_sqrt, args=(lin_eta, ), tol=1e-10, \
                maxiter=1000)


            if i_iter == 0:
                ax1.scatter(lin_eta, root+2*lambda_sqrt_end, s=20, \
                    c=color_n[i_n], label="$n=$ "+str(i_n+1))
            else:
                ax1.scatter(lin_eta, root, s=1, c=color_n[i_n])
            #
        #
    #
#

ax1.grid()
ax1.set_axisbelow(True)

ax1.set_xlim(0, 1)
ax1.set_ylim(lambda_sqrt_init, lambda_sqrt_end)

ax1.set_xlabel("$\\eta = R_\\mathrm{I}/R_\\mathrm{O}$", fontsize=16)
ax1.set_ylabel( "$\\sqrt{\\lambda_\\alpha} = "+
    "\\sqrt{-\\sigma_\\alpha R_\\mathrm{O}^2/\\eta_\\mathrm{m}}$", fontsize=16)
fig1.suptitle(fig_title, color="magenta", fontsize=14)

leg1 = ax1.legend(loc='upper left', fontsize=14)
leg1.get_frame().set_alpha(1)

ax1.tick_params(labelsize=12)

fig1.tight_layout()
fig1.savefig(fig_name, dpi=1000)

# end time
now = datetime.datetime.now()
print('end:   ', now)

plt.show()
