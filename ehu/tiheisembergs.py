import numpy as np

# ---------------------------------------------------------------------
# TRANSLATIONALY INVARIANT XXZ MODEL
# https://arxiv.org/abs/2004.12757
# ---------------------------------------------------------------------

class tincxxz:
   """Class representing the particle-number & tranlationally invariant basis for 
      the XXZ-Hamiltonian with periodic boundary conditions
      
      Objects of this class represent a block diagonal sector of the Hamiltonian
      in this basis"""
   def __init__(self, N: int, n: int, k:int):
      """Arguments
         ---------
         N: Length of the chain
         n: Particle quantum number
         k: Quasimomentum quantum number
      """
      if n >= N or n < 0:
         raise ValueError("n must fulfill 0 ≤ n < N")
      if k >= N or k < 0:
         raise ValueError("k must fulfill 0 ≤ k < N")
      self._base_transform = (2**(np.arange(N)[::-1])).astype(np.uint64)
      self.n = n
      self.N = N
      self.k = k
      if n == 0:
         self.basis_states = [0]
         self.basis_states_period = {0: 1}
         return
      if n == N:
         self.basis_states = [np.sum(np.ones(N, np.uint64)*self._base_transform)]
         self.basis_states_period = {0: 1}
         return
      basis_states_ints = set()
      basis_states_periods = {}
      for period, min_roll in self.number_conserving_ints():
         if np.isclose(k*period//self.N, k*period / self.N):
            basis_states_ints.add(min_roll)
            basis_states_periods[min_roll] = period
      self.basis_states = np.array(list(basis_states_ints), dtype=int)
      self.basis_states_period = basis_states_periods

   @classmethod
   def many_k(cls, N:int, n: int, k_it: int):
      """Returns many tincxxz objects with the same N, n and different k's in k_it"""
      to_return = [cls(N, 0, k) for k in k_it]
      for i in range(len(k_it)):
         to_return[i].n = n
         to_return[i].basis_states = set()
         to_return[i].basis_states_period = {}
      basis_states_ints_k = {k:set() for k in k_it}
      basis_states_periods_k = {k:{} for k in k_it}
      for period, min_roll in to_return[0].number_conserving_ints():
         for k in k_it:
            if np.isclose(k*period//N, k*period/N):
               basis_states_ints_k[k].add(min_roll)
               basis_states_periods_k[k][min_roll] = period
      for i, k in enumerate(k_it):
         to_return[i].basis_states = np.array(list(basis_states_ints_k[k]), dtype=int)
         to_return[i].basis_states_period = basis_states_periods_k[k]
      return to_return
   
   @property
   def dimension(self):
      return len(self.basis_states)
   
   def y(self, state: int):
      """Returns the norm of the state in the basis labeled by its integer representation"""
      return np.sqrt(self.basis_states_period[state])/self.N
   
   def basisstr(self, state_ind=None):
      """Returns a string representation of the given state labeled by its position in the basis.
         If state_ind == None, return a list with all the states in the basis"""
      if state_ind is None:
         return [f"|{i:0{self.N}b}>" for i in self.basis_states]
      else:
         return f"|{self.basis_states[state_ind]:0{self.N}b}>"

   def int_period(self, n, return_min_roll=False):
      """Returns the period the binary string associated with int n"""
      n_bin = np.array(list(f"{n:0{self.N}b}"), dtype=int)
      min_roll = n
      period = self.N
      for i in range(1, self.N):
         roll_n = np.sum(np.roll(n_bin, i)*self._base_transform).astype(int)
         if roll_n < min_roll:
            min_roll = roll_n
         if (roll_n == n) and (period == self.N):
            period = i
            break
      return (period, min_roll) if return_min_roll else period
   
   def int_rolls(self, n):
      n_bin = np.array(list(f"{n:0{self.N}b}"), dtype=int)
      rolls = [n]
      for i in range(1, self.N):
         roll = np.sum(np.roll(n_bin, i)*self._base_transform).astype(int)
         rolls.append(roll)
         if roll == n:
            break
      return np.sort(rolls)

   def number_conserving_ints(self):
      """Generator for all configurations with fixed particle number"""
      min_int = np.sum(np.ones(self.n, np.uint64)*self._base_transform[self.N-self.n::])
      max_int = np.sum(np.concatenate((np.ones(self.n), np.zeros(self.N-self.n)))*self._base_transform)
      yield 0, min_int
      current = min_int
      while (current := next_lex_permutation(current)) <= max_int:
         period, current_min_roll = self.int_period(current, return_min_roll=True)
         yield period, current_min_roll
   
   def _hamiltonian_action(self, global_neg: bool, delta: float, n: int):
      """Returns a dictionary with the coefficients of the action of the XXZ Hamiltonian over the state n
         in the basis labeled by their integer representation."""
      result_dict = {}
      n_bin = np.array(list(f"{n:0{self.N}b}"), dtype=int)
      sign = (-1)**global_neg
      diag = sign*delta*np.sum((2*n_bin - 1)*(2*np.roll(n_bin, 1) - 1))
      result_dict[n] = diag
      for l in range(self.N):
         if n_bin[l] != n_bin[(l+1) % self.N]:
            m_bin = n_bin.copy()
            m_bin[l] = (m_bin[l] + 1) % 2
            m_bin[(l+1) % self.N] = (m_bin[(l+1) % self.N] + 1) % 2
            m = np.sum(m_bin*self._base_transform).astype(int)
            result_dict[m] = 2*sign
      return result_dict
   
   def block_hamiltonian(self, global_neg: bool, delta: float):
      """Returns the block sector of the XXZ hamiltonian for the given n,k."""
      result = np.zeros((len(self.basis_states), len(self.basis_states)), dtype=complex)
      for col_ind, n in enumerate(self.basis_states):
         hamiltonian_action = self._hamiltonian_action(global_neg, delta, n)
         for m, h in hamiltonian_action.items():
            m_bin = np.array(list(f"{m:0{self.N}b}"), dtype=int)
            norm_m = m
            req_shift = 0
            for i in range(1, self.N):
               this_m = np.sum(np.roll(m_bin, i)*self._base_transform).astype(int)
               if this_m == norm_m:
                  break
               if this_m < norm_m:
                  norm_m = int(this_m)
                  req_shift = i
            if norm_m in self.basis_states:
               row_ind = np.argwhere(np.equal(norm_m, self.basis_states))[0][0]
               result[row_ind, col_ind] = result[row_ind, col_ind] + self.y(n)/self.y(norm_m)*h*np.exp(2j*np.pi*req_shift*self.k/self.N)
      return result
   
   def minimum_energy_vec(self, global_neg: bool, delta: float, return_E=False):
      evals, evecs = np.linalg.eigh(self.block_hamiltonian(global_neg, delta))
      return (evals[0], evecs[:, 0]) if return_E else evecs[:, 0]
   
   def change_to_comp_basis(self, state: np.ndarray[complex]):
      result = np.zeros(2**self.N, complex)
      for i, n in enumerate(self.basis_states):
         indices_to_add = self.int_rolls(n)
         result[indices_to_add] += state[i]*np.sqrt(self.basis_states_period[n])/self.N
      return result

   def __len__(self):
      return len(self.basis_states)

   def __repr__(self) -> str:
      return f"<XXZ basis: N={self.N} n={self.n} k={self.k}>"

def next_lex_permutation(n: int):
   """Returns the integer corresponding to the next lexicographic permutation
      of n in binary representation (5 = "101" -> 6 = "110" -> 9 = "1001" -> ...)"""
   n = np.uint32(n)
   t = (n | (n - 1)) + 1
   return t | ((((t & -t) // (n & -n)) >> 1) - 1)