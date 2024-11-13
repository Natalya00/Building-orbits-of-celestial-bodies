import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 2.95912208286e-4
m0 = 1.0
m1 = 3.00348959632E-6
m2 = m1 * 1.23000383E-2

q0 = np.array([
    [0, 0, 0],
    [-0.1667743823220, 0.9690675883429, -0.0000342671456],
    [-0.1694619061456, 0.9692330175719, -0.0000266725711]
])

v0 = np.array([
    [0, 0, 0],  # Sun
    [-0.0172346557280, -0.0029762680930, -0.0000004154391],
    [-0.0172817331582, -0.0035325102831, 0.0000491191454]
])

p0 = np.array([
    v0[0] * m0,
    v0[1] * m1,
    v0[2] * m2
])

def vecf(v):
    return v / np.linalg.norm(v) ** 3

def fun_v(q):
    f = np.zeros((3, 3))
    f[0] = -G * m0 * m1 * vecf(q[0] - q[1]) - G * m0 * m2 * vecf(q[0] - q[2])
    f[1] = -G * m1 * m0 * vecf(q[1] - q[0]) - G * m1 * m2 * vecf(q[1] - q[2])
    f[2] = -G * m2 * m0 * vecf(q[2] - q[0]) - G * m2 * m1 * vecf(q[2] - q[1])
    return f

def fun_u(p):
    return np.array([p[0] / m0, p[1] / m1, p[2] / m2])

def euler(n, h, p, q):
    vq = [q.copy()]
    vp = [p.copy()]
    distances = []
    for _ in range(n):
        q += h * fun_u(p)
        p += h * fun_v(q)
        vq.append(q.copy())
        vp.append(p.copy())
        distances.append(np.linalg.norm(q[2] - q[0]))
    return np.array(vp), np.array(vq), np.array(distances)

def runge_kutta_4(n, h, p, q):
    vq = [q.copy()]
    vp = [p.copy()]
    distances = []
    for _ in range(n):
        k1_q = h * fun_u(p)
        k1_p = h * fun_v(q)

        k2_q = h * fun_u(p + 0.5 * k1_p)
        k2_p = h * fun_v(q + 0.5 * k1_q)

        k3_q = h * fun_u(p + 0.5 * k2_p)
        k3_p = h * fun_v(q + 0.5 * k2_q)

        k4_q = h * fun_u(p + k3_p)
        k4_p = h * fun_v(q + k3_q)

        q += (k1_q + 2 * k2_q + 2 * k3_q + k4_q) / 6
        p += (k1_p + 2 * k2_p + 2 * k3_p + k4_p) / 6

        vq.append(q.copy())
        vp.append(p.copy())

        distances.append(np.linalg.norm(q[2] - q[0]))
    return np.array(vp), np.array(vq), np.array(distances)

def dormand_prince(t_span, h0, p, q, tol=1e-6):
    vq = [q.copy()]
    vp = [p.copy()]
    distances = []

    t = 0
    h = h0
    while t < t_span:
        k1_q = h * fun_u(p)
        k1_p = h * fun_v(q)

        k2_q = h * fun_u(p + k1_p * 0.2)
        k2_p = h * fun_v(q + k1_q * 0.2)

        k3_q = h * fun_u(p + k1_p * (3 / 40) + k2_p * (9 / 40))
        k3_p = h * fun_v(q + k1_q * (3 / 40) + k2_q * (9 / 40))

        k4_q = h * fun_u(p + k1_p * (44 / 45) - k2_p * (56 / 15) + k3_p * (32 / 9))
        k4_p = h * fun_v(q + k1_q * (44 / 45) - k2_q * (56 / 15) + k3_q * (32 / 9))

        k5_q = h * fun_u(p + k1_p * (19372 / 6561) - k2_p * (25360 / 2187) + k3_p * (64448 / 6561) - k4_p * (212 / 729))
        k5_p = h * fun_v(q + k1_q * (19372 / 6561) - k2_q * (25360 / 2187) + k3_q * (64448 / 6561) - k4_q * (212 / 729))

        k6_q = h * fun_u(p + k1_p * (9017 / 3168) - k2_p * (355 / 33) + k3_p * (46732 / 5247) + k4_p * (49 / 176) - k5_p * (5103 / 18656))
        k6_p = h * fun_v(q + k1_q * (9017 / 3168) - k2_q * (355 / 33) + k3_q * (46732 / 5247) + k4_q * (49 / 176) - k5_q * (5103 / 18656))

        k7_q = h * fun_u(p + k1_p * (35 / 384) + k3_p * (500 / 1113) + k4_p * (125 / 192) - k5_p * (2187 / 6784) + k6_p * (11 / 84))
        k7_p = h * fun_v(q + k1_q * (35 / 384) + k3_q * (500 / 1113) + k4_q * (125 / 192) - k5_q * (2187 / 6784) + k6_q * (11 / 84))

        q_next = q + (35 / 384 * k1_q + 500 / 1113 * k3_q + 125 / 192 * k4_q - 2187 / 6784 * k5_q + 11 / 84 * k6_q)
        p_next = p + (35 / 384 * k1_p + 500 / 1113 * k3_p + 125 / 192 * k4_p - 2187 / 6784 * k5_p + 11 / 84 * k6_p)

        q_err = (q_next - (q + 5179 / 57600 * k1_q + 7571 / 16695 * k3_q + 393 / 640 * k4_q - 92097 / 339200 * k5_q + 187 / 2100 * k6_q + 1 / 40 * k7_q))
        p_err = (p_next - (p + 5179 / 57600 * k1_p + 7571 / 16695 * k3_p + 393 / 640 * k4_p - 92097 / 339200 * k5_p + 187 / 2100 * k6_p + 1 / 40 * k7_p))
        error = np.sqrt(np.sum(q_err ** 2) + np.sum(p_err ** 2))

        if error > tol:
            h = h * 0.9 * (tol / error) ** 0.2
            continue

        q = q_next
        p = p_next
        t += h

        if error < tol / 10:
            h = min(h * 2, h0)

        vq.append(q.copy())
        vp.append(p.copy())
        distances.append(np.linalg.norm(q[2] - q[0]))

    return np.array(vp), np.array(vq), np.array(distances)

exp = 1
time_span = int(365 / exp)
h = 1 * exp
#vp, vq, distances = euler(time_span, h, p0, q0)
#vp, vq, distances = runge_kutta_4(time_span, h, p0, q0)
vp, vq, distances = dormand_prince(time_span, h, p0, q0)

noise = np.random.normal(0, 0.01, distances.shape)
distances_noisy = distances + noise

vq[:, 1, :] -= vq[:, 0, :]
vq[:, 2, :] -= vq[:, 0, :]
vq[:, 2, :] = vq[:, 1, :] + 100 * (vq[:, 2, :] - vq[:, 1, :])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.plot([0], [0], [0], 'yo', markersize=10, label="Солнце")
earth_line, = ax.plot([], [], [], 'b', label="Земля")
moon_line, = ax.plot([], [], [], 'r--', label="Луна")
earth_point, = ax.plot([], [], [], 'bo')
moon_point, = ax.plot([], [], [], 'ro')

def init():
    earth_line.set_data([], [])
    earth_line.set_3d_properties([])
    moon_line.set_data([], [])
    moon_line.set_3d_properties([])
    earth_point.set_data([], [])
    earth_point.set_3d_properties([])
    moon_point.set_data([], [])
    moon_point.set_3d_properties([])
    return earth_line, moon_line, earth_point, moon_point

def update(num):
    earth_line.set_data(vq[:num, 1, 0], vq[:num, 1, 1])
    earth_line.set_3d_properties(vq[:num, 1, 2])

    moon_line.set_data(vq[:num, 2, 0], vq[:num, 2, 1])
    moon_line.set_3d_properties(vq[:num, 2, 2])

    earth_point.set_data([vq[num, 1, 0]], [vq[num, 1, 1]])
    earth_point.set_3d_properties([vq[num, 1, 2]])

    moon_point.set_data([vq[num, 2, 0]], [vq[num, 2, 1]])
    moon_point.set_3d_properties([vq[num, 2, 2]])

    return earth_line, moon_line, earth_point, moon_point

k = max(1, int(0.1 / h))
ani = FuncAnimation(fig, update, frames=range(0, len(vq), k), init_func=init, interval=10, blit=True)

plt.legend()
plt.show()

plt.figure()
time = np.arange(len(distances)) * h * 0.1  # Временная шкала в днях
plt.plot(time, distances, label="Расстояние между Солнцем и Луной без наложения шума")
plt.plot(time, distances_noisy, label="Расстояние между Солнцем и Луной с наложением шума", linestyle='--')
plt.xlabel("Время (дни)")
plt.ylabel("Расстояние (AU)")
plt.legend()
plt.show()