"""
check for eigenvalue problem

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

# Tex
from matplotlib import rc
rc('text', usetex=True)

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
eta = 0.3

# degree
n = 5
" === end [parameter] === "

fig1 = plt.figure( figsize=(6, 6) )
ax1 = fig1.add_subplot(111)

# toroidal free decay mode & poloidal free decay mode (Ipc/Opc, FS/Opc)
if (switch_tp == 0) or \
    ( (switch_tp == 1) and (switch_obc == 1) and (switch_ibc == 1) ):

    fig_title = "Toroidal free decay mode / "+ \
        "Poloidal free decay mode (Ipc/Opc, FS/Opc)"

    x_test = np.linspace(0.01, 30, 1000)

    # full sphere
    f_fs = jv(n+1/2, x_test)
    ax1.plot(x_test, f_fs, c="red", label="FS, $n =$ "+str(n))

    # spherical shell
    f = jv(n+1/2, eta * x_test) \
        * yv(n+1/2, x_test) \
        - jv(n+1/2, x_test) \
        * yv(n+1/2, eta * x_test)
    ax1.plot(x_test, f, c="blue", label="SS, $n =$ "+str(n)+ \
        ", $\\eta =$ "+str(eta))
#

# poloidal free decay mode
if switch_tp == 1:

    # Ii/Oi
    if (switch_obc == 0) and (switch_ibc == 0):
        fig_title = "Poloidal free decay mode (Ii/Oi)"
    # Ii/Opc
    elif (switch_obc == 1) and (switch_ibc == 0):
        fig_title = "Poloidal free decay mode (Ii/Opc)"
    # Ipc/Oi
    elif (switch_obc == 0) and (switch_ibc == 1):
        fig_title = "Poloidal free decay mode (Ipc/Oi)"
    #

    x_test = np.linspace(0.01, 30, 1000)

    # full sphere
    # Ii/Opc
    if (switch_obc == 1) and (switch_ibc == 0):
        f_fs = jv(n+1/2, x_test)
    # Ii/Oi, Ipc/Oi
    else:
        f_fs = jv(n-1/2, x_test)
    #
    ax1.plot(x_test, f_fs, c="red", label="FS, $n =$ "+str(n))

    # spherical shell
    # Ii/Oi
    if (switch_obc == 0) and (switch_ibc == 0):
        f = ( (2*n+1) * jv(n+1/2, eta * x_test) / ( eta * x_test ) \
            - jv(n-1/2, eta * x_test) ) \
            * yv(n-1/2, x_test) \
            - ( (2*n+1) * yv(n+1/2, eta * x_test) / ( eta * x_test ) \
            - yv(n-1/2, eta * x_test) ) \
            * jv(n-1/2, x_test)
    # Ii/Opc
    elif (switch_obc == 1) and (switch_ibc == 0):
        f = ( (2*n+1) * jv(n+1/2, eta * x_test) / ( eta * x_test ) \
            - jv(n-1/2, eta * x_test) ) \
            * yv(n+1/2, x_test) \
            - ( (2*n+1) * yv(n+1/2, eta * x_test) / ( eta * x_test ) \
            - yv(n-1/2, eta * x_test) ) \
            * jv(n+1/2, x_test)
    # Oi/Ipc
    elif (switch_obc == 0) and (switch_ibc == 1):
        f = jv(n+1/2, eta * x_test) \
            * yv(n-1/2, x_test) \
            - jv(n-1/2, x_test) \
            * yv(n+1/2, eta * x_test)
    #
    ax1.plot(x_test, f, c="blue", label="SS, $n =$ "+str(n)+ \
        ", $\\eta =$ "+str(eta))
    #
#

ax1.grid()
ax1.set_axisbelow(True)

ax1.set_ylim(-0.5, 1)

ax1.set_xlabel( "$\\sqrt{\\lambda_\\alpha}$ $($if $ y = 0)$", fontsize=16)
fig1.suptitle(fig_title, color="magenta", fontsize=14)

leg1 = ax1.legend(loc='upper right', fontsize=14)
leg1.get_frame().set_alpha(1)

ax1.tick_params(labelsize=12)

fig1.tight_layout()

plt.show()
