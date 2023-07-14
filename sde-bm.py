import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

np.random.seed(0)

T, N = 1, 500
dt = T / N

# Number of realizations to generate.
m = 1000

# Create an empty array to store the realizations.
W = np.empty((m, N+1))

# Initial values of x.
W[:, 0] = 0

dW = np.random.normal(size=(m, N), loc=0, scale=np.sqrt(dt))
W[:, 1:] = np.cumsum(dW, axis=-1) + np.expand_dims(W[:, 0], axis=-1)

t = np.linspace(0, N * dt, N + 1)

plt.figure(figsize=(8, 4))
for i in range(10):
    plt.plot(t, W[i])
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$W_t$')
print(f'T = {T}, N = {N}, dt = {dt:.5f}')
plt.savefig('proj/sde-bm.pdf')
plt.show()

###### U_mean approx. ######

U_true = np.exp(9 * t / 8)

U1 = np.exp(t + 0.5 * W[:10, ])
U2 = np.exp(t + 0.5 * W[:100, ])
U3 = np.exp(t + 0.5 * W)

U_mean1 = U1.mean(axis=0)
U_mean2 = U2.mean(axis=0)
U_mean3 = U3.mean(axis=0)

averr1 = np.linalg.norm(U_mean1 - U_true, ord=np.inf)
averr2 = np.linalg.norm(U_mean2 - U_true, ord=np.inf)
averr3 = np.linalg.norm(U_mean3 - U_true, ord=np.inf)

plt.figure(figsize=(8, 4))
plt.plot(t, U1[0], linestyle='--', color='grey', label=f'5 sample paths')
for i in range(4):
    plt.plot(t, U1[i+1], linestyle='--', color='grey')
plt.plot(t, U_mean1, label=f'avg. over {U1.shape[0]} paths (error = {averr1:.3f})')
plt.plot(t, U_mean2, label=f'avg. over {U2.shape[0]} paths (error = {averr2:.3f})')
plt.plot(t, U_mean3, label=f'avg. over {U3.shape[0]} paths (error = {averr3:.3f})')
plt.plot(t, U_true, 'k', label=f'theoretical expected values: $\exp(9t/8)$')
plt.grid()
plt.xlabel('$t$')
plt.ylabel('$u(t, W_t)$')
plt.legend()
plt.savefig('proj/sde-bm-u_mean.pdf')
plt.show()