import numpy as np
import mdptoolbox.mdp as mdp
import mdptoolbox.example as mdp_exp


def value_iteration(S, A, P, gamma=.99, theta=0.0000001):
    V = np.random.random(len(S))
    for i in range(10000):
        old_V = V.copy()
        Q = np.zeros((len(S), len(A)), dtype=float)
        for s in S:
            for a in A:
                if s in P:
                    if a in P[s]:
                        for prob, s_prime, reward, done in P[s][a]:
                            if s < len(old_V) and s_prime < len(old_V):
                                sa = old_V[s_prime]
                                Q[s][a] += prob * (reward + gamma * old_V[s_prime] * (not done))
            if s < len(old_V):
                V[s] = Q[s].max()
        if np.all(np.abs(old_V - V) < theta):
            break

    pi = np.argmax(Q, axis=1)
    return pi, V


def create_transition_matrix(is_bad_side, num_rolls):
    P = {}
    N = len(is_bad_side)
    next_state_bankrolls = [0]
    for roll in range(1, num_rolls+1):
        idx = 0
        normalized_denom = N ** roll
        next_state_bankroll_set = []
        while idx < len(next_state_bankrolls):
            bankroll = next_state_bankrolls[idx]
            next_prob_states, is_terminal = prob_next_state(N, normalized_denom, bankroll, next_state_bankroll_set)
            if is_terminal:
                return P
            action = action_set(next_prob_states, bankroll)
            P[bankroll] = action.copy()
            idx += 1

        next_state_bankrolls = list(set(next_state_bankroll_set))
    return P


def prob_next_state(N, normalized_denom, bankroll, next_state_bankroll_set):
    next_prob_states = []
    reward_prob_sum = 0
    for i in range(0, N):
        if is_bad_side[i] == 1:
            next_prob_states.append((1 / normalized_denom, 0, bankroll*-1, True))
            reward_prob_sum += (1 / normalized_denom) * (bankroll*-1)
        else:
            reward = i + 1
            next_state = bankroll + reward
            next_prob_states.append((1 / normalized_denom, next_state, reward, False))
            reward_prob_sum += (1 / normalized_denom) * reward
            next_state_bankroll_set.append(next_state)
    if reward_prob_sum <= 0:
        return next_prob_states, True
    return next_prob_states, False


def action_set(next_prob_states, bankroll):
    action = {0: next_prob_states}
    quit_game = [(1, bankroll, 0, True)]
    action[1] = quit_game
    return action


is_bad_side = [1, 1, 1, 0, 0, 0]
# is_bad_side = [1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0]
# is_bad_side = [1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0]
num_rolls = 2
P = create_transition_matrix(is_bad_side, num_rolls)
print(P)
S = list(range(0, 7))
A = list(range(0, 2))

pi, v = value_iteration(S, A, P, gamma=1)
print(np.mean(v))
print(v)
print(pi)