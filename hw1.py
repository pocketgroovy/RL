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
                for prob, s_prime, reward, done in P[s][a]:
                    if s < len(old_V) and s_prime < len(old_V):
                        Q[s][a] += prob * (reward + gamma * old_V[s_prime] * (not done))
            if s < len(old_V):
                V[s] = Q[s].max()
        if np.all(np.abs(old_V - V) < theta):
            break

    pi = np.argmax(Q, axis=1)
    return pi, V


# def value_iteration(P, R, gamma=.99, theta=0.0000001):
#     vi = mdp.ValueIteration(P, R, discount=gamma)
#     return vi


# S = range(6)
# A = range(2)
# ## {action
# P = {0:
#      {0:
#       [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 0, 0.0, False),(0.3333333333333333, 4, 0.0, False)],
#       1:
#       [(0.3333333333333333, 0, 0.0, False), (0.3333333333333333, 4, 0.0, False),
#           (0.3333333333333333, 1, 0.0, False)]}}

# pi, V = value_iteration(S, A, P)

# P, R = mdp_exp.forest(S=6, p=0.167)
# vi = value_iteration(P, R)
# print(vi)
# vi.run()
# print(vi.policy)


def create_transition_matrix(is_bad_side, num_states):
    P = {}
    action = {}
    size = len(is_bad_side)
    next_state_bankroll = []
    next_state_bankroll.append(0)
    for j in range(0, num_states):
        idx = 0
        next_state_bankroll_set = []
        while idx < len(next_state_bankroll):
            bankroll = next_state_bankroll[idx]
            next_prob_states = []
            quit_game = []

            # possibility of next state
            for i in range(0, size):
                if is_bad_side[i] == 1:
                    next_prob_states.append((1 / size, 0, 0, True))
                else:
                    reward = i + 1
                    next_state = bankroll + reward
                    next_prob_states.append((1 / size, next_state, reward, False))
                    next_state_bankroll_set.append(next_state)
            action[0] = next_prob_states
            quit_game.append((1, bankroll, 0, True))
            action[1] = quit_game
            P[bankroll] = action.copy()
            idx += 1
        next_state_bankroll = list(set(next_state_bankroll_set))
    return P

# def prob_next_state(size):



is_bad_side = [1, 1, 1, 0, 0, 0]
# is_bad_side = [1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,1,0]
# is_bad_side = [1,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,0,1,0]
num_states = 2
P = create_transition_matrix(is_bad_side, num_states)
print(P)
S = P.keys()
A = list(range(0, 2))

pi, v = value_iteration(S, A, P, gamma=0.95)
print(np.mean(v))