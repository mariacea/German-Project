from sunny import analytical_shannon as ashannon
import scipy as sp
import numpy as np
import tiheisembergs
import os

def get_amps(basis, global_neg, delta):
    hamiltonian = basis.block_hamiltonian(global_neg, delta)
    evals, evecs = sp.sparse.linalg.eigsh(hamiltonian, which="SA", k=1)
    sort_mask = np.argsort(evals)
    Egs = evals[sort_mask[0]]
    comp_basis_state = basis.change_to_comp_basis(evecs[:, sort_mask[0]])
    amps = np.abs(comp_basis_state)**2
    return Egs, amps

def get_xxz_numerical_max_amps(N_arr, delta_arr, global_neg, return_energy=False, return_max_amp_ind=False, return_cnum=False, comparison_tol=1e-8, print_mode=False, save=True):
    N_arr = np.asarray(N_arr)
    if N_arr.ndim == 0:
        N_arr = N_arr[None]
    delta_arr = np.asarray(delta_arr)
    if delta_arr.ndim == 0:
        delta_arr = delta_arr[None]
    max_prob_num_arr = np.zeros((len(N_arr), len(delta_arr)), dtype=float)
    if return_energy: Egs_num_arr = np.zeros_like(max_prob_num_arr, dtype=float)
    if return_max_amp_ind: max_amp_inds_arr = np.zeros_like(max_prob_num_arr, dtype=int)
    if return_cnum: n_max_conf_num_arr = np.zeros_like(max_prob_num_arr, dtype=int)
    for i, N in enumerate(N_arr):
        gs_ks = [0] if global_neg else [(N - N // 2 - 1)//2 + (N - N // 2 - 1) % 2, N - (N - N // 2 - 1)//2 - (N - N // 2 - 1) % 2]
        bases = None
        for j, delta in enumerate(delta_arr):
            if print_mode: print(f"\rNum: N = {N} / Δ = {delta:.02f}", end="")
            if save:
                filename = f"Results/amplitudes_xxz_model_N_{N}_negsign_{global_neg}_delta_{delta:.04f}.npy"
            else:
                filename = ""
            if not os.path.exists(filename):
                if (bases is None) or (bases[0].N != N):
                    bases = tiheisembergs.tincxxz.many_k(N, N//2, gs_ks)
                Egs, amps = get_amps(bases[0], global_neg, delta)
                filedata = np.zeros(2**N + 1)
                filedata[0] = Egs
                filedata[1::] = amps
                np.savetxt(filename, filedata)
            else:
                filedata = np.loadtxt(filename)
                Egs = filedata[0]
                amps = filedata[1::]
            max_amp_ind = np.argmax(amps)
            max_prob_num_arr[i, j] = amps[max_amp_ind]
            if return_energy: Egs_num_arr[i, j] = Egs
            if return_max_amp_ind: max_amp_inds_arr[i, j] = max_amp_ind
            if return_cnum: n_max_conf_num_arr[i, j] = np.sum(np.isclose(amps, amps[max_amp_ind], atol=comparison_tol))
    to_return = [np.squeeze(max_prob_num_arr)]
    if return_energy: to_return.append(np.squeeze(Egs_num_arr))
    if return_max_amp_ind: to_return.append(np.squeeze(max_amp_inds_arr))
    if return_cnum: to_return.append(np.squeeze(n_max_conf_num_arr))
    return to_return if len(to_return) > 1 else to_return[0]

def get_xxz_numerical_shannon(N_arr, delta_arr, global_neg, return_energy=False, spin_penalty_factor=10, print_mode=False, save=True):
    N_arr = np.asarray(N_arr)
    if N_arr.ndim == 0:
        N_arr = N_arr[None]
    delta_arr = np.asarray(delta_arr)
    if delta_arr.ndim == 0:
        delta_arr = delta_arr[None]
    shannon_arr = np.zeros((len(N_arr), len(delta_arr)), dtype=float)
    if return_energy: Egs_num_arr = np.zeros_like(shannon_arr, dtype=float)
    for i, N in enumerate(N_arr):
        gs_ks = [0] if global_neg else [(N - N // 2 - 1)//2 + (N - N // 2 - 1) % 2, N - (N - N // 2 - 1)//2 - (N - N // 2 - 1) % 2]
        bases = None
        for j, delta in enumerate(delta_arr):
            if print_mode: print(f"\rNum: N = {N}", end="")
            if save: 
                filename = f"Results/amplitudes_xxz_model_N_{N}_negsign_{global_neg}_delta_{delta:.04f}.npy"
            else:
                filename = None
            if not os.path.exists(filename):
                if (bases is None) or (bases[0].N != N):
                    bases = tiheisembergs.tincxxz.many_k(N, N//2, gs_ks)
                Egs, amps = get_amps(bases[0], global_neg, delta)
                filedata = np.zeros(2**N + 1)
                filedata[0] = Egs
                filedata[1::] = amps
                np.savetxt(filename, filedata)
            else:
                filedata = np.loadtxt(filename)
                Egs = filedata[0]
                amps = filedata[1::]
            non_zero_mask = ~np.isclose(amps, 0)
            shannon_arr[i, j] = -np.sum(amps[non_zero_mask]*np.log(amps[non_zero_mask]))
            if return_energy: Egs_num_arr[i, j] = Egs
    to_return = [np.squeeze(shannon_arr)]
    if return_energy: to_return.append(np.squeeze(Egs_num_arr))
    return to_return if len(to_return) > 1 else to_return[0]

def conf_index_to_str(index, N):
    return f"|{int(index):0{N}b}>"

def get_analytical_xx_max_amps(N_arr, return_energy=False, return_max_amp_ind=False, return_cnum=False, comparison_tol=1e-8, print_mode=False):
    max_prob_num_arr = np.zeros_like(N_arr, dtype=float)
    if return_energy: Egs_num_arr = np.zeros_like(N_arr, dtype=float)
    if return_max_amp_ind: max_amp_ind_arr = np.zeros_like(N_arr)
    if return_cnum: n_max_conf_num_arr = np.zeros_like(N_arr, dtype=float)
    for i, N in enumerate(N_arr):
        if print_mode: print(f"\rAn: N = {N}", end="")
        filename = f"Results/an_amplitudes_xx_model_N_{N}.npy"
        if not os.path.exists(filename):
            confarr = ashannon.probabilities(N)
            amps = [conf.prob for conf in confarr]
            Egs = ashannon.ground_state_energy(N)
            filedata = np.zeros(len(amps) + 1)
            filedata[0] = Egs
            filedata[1::] = amps
            np.save(filename, filedata)
        else:
            filedata = np.load(filename)
            Egs = filedata[0]
            amps = filedata[1::]
        max_amp_ind = np.argmax(amps)
        max_prob_num_arr[i] = amps[max_amp_ind]
        if return_energy: Egs_num_arr[i] = Egs
        if return_max_amp_ind: max_amp_ind_arr[i] = max_amp_ind
        if return_cnum: n_max_conf_num_arr[i] = np.sum(np.isclose(amps, amps[max_amp_ind], atol=comparison_tol))
    to_return = [max_prob_num_arr]
    if return_energy: to_return.append(Egs_num_arr)
    if return_max_amp_ind: to_return.append(max_amp_ind_arr)
    if return_cnum: to_return.append(n_max_conf_num_arr)
    return np.squeeze(to_return)