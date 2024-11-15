# Моделирование орбиты Луны.

---
## Описание
В коде моделируется взаимодействие Солнца, Земли и Луны, используя ньютоновскую механику для расчета орбит. Моделирование позволяет проследить орбитальные траектории и графически представить изменение расстояния между Солнцем и Луной с добавлением случайного шума.

---

## Оглавление
1. Описание кода 
   * Используемые константы
   * Начальные условия
   * Основные функции для расчета физических величин
   * Методы численного интегрирования 
   * Добавление шума
   * Подготовка к визуализации
   * Анимация орбит
   * График расстояния "Солнце — Луна"
2. Математические выкладки
3. Результаты

---

## 1. Описание кода
### Используемые константы
```
G = 2.95912208286e-4  # Гравитационная постоянная
m0 = 1.0  # Масса Солнца
m1 = 3.00348959632E-6  # Масса Земли
m2 = m1 * 1.23000383E-2  # Масса Луны
```
**G** — гравитационная постоянная в астрономических единицах (AU³/год²/масса Солнца).
**m0, m1, m2** — массы Солнца, Земли и Луны. Массы Земли и Луны заданы относительно массы Солнца для удобства расчетов.

### Начальные условия
```
q0 = np.array([
    [0, 0, 0],  # Солнце
    [-0.1667743823220, 0.9690675883429, -0.0000342671456],  # Земля
    [-0.1694619061456, 0.9692330175719, -0.0000266725711]  # Луна
])
v0 = np.array([
    [0, 0, 0],  # Солнце
    [-0.0172346557280, -0.0029762680930, -0.0000004154391],  # Земля
    [-0.0172817331582, -0.0035325102831, 0.0000491191454]  # Луна
])
```
**q0 и v0** — начальные положения и скорости для Солнца, Земли и Луны в астрономических единицах (AU).

### Основные функции для расчета физических величин
#### Нормализация веторов
```
def vecf(v):
    return v / np.linalg.norm(v) ** 3
```
Функция нормализует вектор и делит его на куб его длины, что соответствует физической модели гравитационного притяжения, где сила обратно пропорциональна квадрату расстояния, а для вычисления направления силы используется единичный вектор, направленный по линии соединения объектов.
#### Закон Всемирного тяготения
```
def fun_v(q):
    f = np.zeros((3, 3))
    f[0] = -G * m0 * m1 * vecf(q[0] - q[1]) - G * m0 * m2 * vecf(q[0] - q[2])
    f[1] = -G * m1 * m0 * vecf(q[1] - q[0]) - G * m1 * m2 * vecf(q[1] - q[2])
    f[2] = -G * m2 * m0 * vecf(q[2] - q[0]) - G * m2 * m1 * vecf(q[2] - q[1])
    return f
```
Функция вычисляет гравитационные силы, действующие на Солнце, Землю и Луну, на основе закона всемирного тяготения Ньютона.
Силы рассчитываются как векторные суммы для каждой пары тел (Солнце-Земля, Солнце-Луна, Земля-Луна).
Таким образом, эта функция возвращает массив сил, действующих на каждый объект в системе, и представляет собой динамическую модель взаимодействия между Солнцем, Землей и Луной.

#### Закон движения Ньютона
```
def fun_u(p):
    return np.array([p[0] / m0, p[1] / m1, p[2] / m2])
Функция fun_u возвращает скорости тел на основе их импульсов.
```
Функция использует импульсы, переданные вектору p, и делит их на соответствующие массы для вычисления скоростей тел. Эта функция является практическим применением второго закона Ньютона, который связывает импульс тела с его движением.
Таким образом, функция находит скорости тел на основе их импульсов, что необходимо для обновления их положения на каждом шаге численного интегрирования.

### Методы численного интегрирования
#### Метод симплектического Эйлера
```
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
```
Метод Эйлера обновляет положение тела и импульсы тел, используя текущие значения сил и скоростей.
В каждом шаге позиции обновляются по скоростям, а импульсы — по силам.
Метод подходит для симуляций с сохранением энергии, но имеет ограничения по точности.
В каждом шаге рассчитывается расстояние между Луной и Солнцем и сохраняется в массиве distances. 
Функция np.linalg.norm() вычисляет евклидово расстояние между двумя точками в 3D-пространстве.

#### Метод Рунге-Кутты 4-го порядка
Функция runge_kutta_4 реализует численный метод интегрирования системы дифференциальных уравнений второго порядка, используя метод Рунге-Кутты 4-го порядка. Этот метод является более точным по сравнению с методом Эйлера, потому что он учитывает информацию о поведении системы не только в текущем времени, но и в нескольких промежуточных точках (на каждом шаге).
Происходит обновление позиций и импульсов объектов (Солнце, Земля, Луна) на каждом шаге с использованием четырех промежуточных вычислений (k1, k2, k3, k4).  
k1: Начальные вычисления на текущем шаге времени.
k2 и k3: Полуобновления, учитывающие частичное изменение положения и импульса.  
k4: Финальное обновление с полным учетом изменения за шаг времени.  
Все значения используются для вычисления среднего значения, что дает более точные обновления положения и импульса.
Позиции (q) и импульсы (p) обновляются с учетом взвешенного среднего значений для каждого шага времени.
На каждом шаге вычисляется новое расстояние между Луной и Солнцем, которое сохраняется в массив distances.


### Метод Дормана-Принса
Метод Дормана-Принса представляет собой адаптивный метод, использующий серию промежуточных значений для получения следующего 
значения решения, а также для оценки ошибки на каждом шаге. Он включает семь стадий (коэффициентов
k) для расчета следующего значения, и это позволяет адаптивно изменять шаг в зависимости от ошибки.  
шаг 1 (k1_q и k1_p): базовый шаг, исходя из текущих p и q.  
шаг 2-6 (k2_q и k2_p и т.д.): промежуточные коэффициенты, которые учитывают все предыдущие стадии. Они содержат коэффициенты, которые подбирают вес вклада каждой стадии, чтобы получить более точное решение.  
шаг 7 (k7_q и k7_p): последняя стадия, рассчитанная с использованием предыдущих шагов, позволяет оценить погрешность шага.  
q_next и p_next (новые значения положения и импульса) - взвешенная сумма k_1–k_6  
Погрешность на текущем шаге оценивается сравнением двух решений:  
Основное решение с учетом всех коэффициентов.  
Контрольное решение на основе дополнительных весов для проверки точности.  
Оценка ошибки error показывает, насколько сильно отличаются значения, полученные двумя разными способами. Если ошибка выше допустимой tol, шаг уменьшается, иначе – принимается, и t увеличивается на h.  
Если шаг был принят, обновляются значения q и p, добавляются текущие значения в массивы vq и vp, и рассчитывается расстояние между Солнцем и Луной для записи в distances.
#### Добавление шума
```
noise = np.random.normal(0, 0.01, distances.shape)
distances_noisy = distances + noise
```
Добавляется случайный шум к расстояниям между Солнцем и Луной. Шум моделирует возможные ошибки наблюдений.

### Подготовка к визуализации
#### Определение параметров
```
exp = 1e-1
time_span = int(365 / exp)
h = 1 * exp
```
**time_span** — это количество шагов (итераций) в расчёте. (временной промежуток составляет 365 дней)
**h** — это шаг по времени для интегрирования.  

#### Корректировка позиций Земли и Луны относительно Солнца
```
vq[:, 1, :] -= vq[:, 0, :]
vq[:, 2, :] -= vq[:, 0, :]
vq[:, 2, :] = vq[:, 1, :] + 100 * (vq[:, 2, :] - vq[:, 1, :])
```
Солнце фиксируется в начале координат, а Земля и Луна позиционируются относительно него. Для удобства отображения координаты Луны масштабируются таким образом, чтобы орбита Луны стала более заметной.

#### Подготовка анимации и отображение
```
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
```
Создается фигура для 3D-графика с помощью matplotlib. 
Происходит отображение Солнца. Создаются линии для орбит Земли и Луны, которые будут обновляться на каждом шаге анимации.
**earth_point и moon_point** — создаются точки, которые будут показывать текущие положения Земли и Луны в каждый момент времени.

#### Инициализация и обновление анимации
```
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
```
Функция выполняет инициализацию анимации, устанавливая начальные значения для линий и точек (пустые списки), чтобы анимация начиналась с пустым графиком.
```
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
```
Функция обновляет графики тел и их орбит на каждом шаге анимации, используя значения, полученные в результате интегрирования.

### Анимация орбит
```
ani = FuncAnimation(fig, update, frames=range(0, len(vq), k), init_func=init, interval=10, blit=True)
```
Используется библиотека FuncAnimation для анимации движения Земли и Луны в 3D.
В каждой итерации анимации обновляется положение Земли и Луны для визуализации их орбит.
**update** — функция, которая будет вызываться для обновления графиков.  
**frames = range(0, len(vq), k)** — указывает, что анимация будет обновляться для каждого шага, начиная с 0, с шагом k, чтобы снизить частоту обновлений и сделать анимацию плавной.  
**interval = 10** — определяет интервал между обновлениями (в миллисекундах).  
**blit = True** — обновляет только измененные части графика.  

### График расстояния "Солнце — Луна"
```
noise = np.random.normal(0, 0.001, distances.shape)
distances_noisy = distances + noise
```
Добавление шума к данным, моделируя погрешности в измерениях или нестабильности в орбитах. Используется нормальное распределение с средним 0 и стандартным отклонением 0.001.
```
plt.figure()
plt.plot(distances, label="Расстояние между Солнцем и Луной без наложения шума")
plt.plot(distances_noisy, label="Расстояние между Солнцем и Луной с наложения шума", linestyle='--')
plt.xlabel("Время")
plt.ylabel("Расстояние (AU)")
plt.legend()
plt.show()
```
График отображает изменение расстояния между Солнцем и Луной с учетом случайного шума.
Две линии: одна для истинного расстояния, другая для зашумленного, показывают влияние ошибок на наблюдение расстояния.

----

# 2. Математические выкладки

## Основные физические законы

### Закон всемирного тяготения: 
Один из основополагающих законов небесной механики, который описывает взаимодействие между двумя телами. Закон Ньютона гласит, что сила гравитационного взаимодействия между двумя телами:
_F = (G * m1 * m2) / r^2_  
G используется в астрономических единицах (AU).

### Второй закон Ньютона (уравнение движения): 
Второй закон Ньютона описывает изменение импульса тела под действием внешних сил:
**_F =  m * a_**

Для моделирования движения тел используются уравнения Ньютона, которые записываются для каждого тела (Солнце, Земля, Луна) с учетом гравитационных взаимодействий между ними. Для двух тел сила гравитационного взаимодействия записывается как:
**_F12 = -F21 = -G * u ((m1 * m2) / (r12)^2)_**.
Таким образом, для каждого тела вычисляются силы, действующие на них со стороны других тел. На основе этих сил затем вычисляется ускорение и изменение скорости с течением времени.
**m1, m2** - массы первого и второго тела соответственно;  
**r12** - расстояние между телами;  
**u** - единичный вектор от тела 1 к телу 2;

Ньютон дает обоснование законам Кеплера: они описывают движение тела, подчиняющегося единственной силе притяжения - силе притяжения Солнца. Идея состоит в том, чтобы применять силу притяжения Солнца не непрерывно во времени, а в виде последовательности импульсов. 
Пусть S — это положение Солнца, и пусть планета изначально находится в точке A. Сначала предположим, что Солнце вообще не 
оказывает никакой гравитационной силы; в этом случае планета движется в течение некоторого времени от точки A до 
точки B по прямой линии с постоянной скоростью в направлении AB. Подождав такое же время, планета должна продолжать движение 
по той же прямой линии до точки C, где вектор AB равен вектору Bc.  
Однако теперь мы прикладываем импульс силы от Солнца к планете: эта сила добавляет компоненту скорости к движению планеты, 
которую Ньютон представляет вектором BV вдоль отрезка SB. Скорость планеты теперь складывается из двух компонент: 
вектора Bc и вектора BV, и результирующий вектор равен BC, который определяет новую точку C. Планета, таким образом, 
движется с этой новой постоянной скоростью до тех пор, пока не достигнет точки C. Повторяя этот процесс, планета следует 
по пути A, B, C, D, E, F и так далее. Ньютон доказывает, что все треугольники SAB, SBC, SCD, SDE, SEF имеют одинаковую площадь

![img.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/img.png)

Рассмотрим систему Солнце–Земля–Луна, где для упрощения мы пренебрегаем другими телами и воздействиями в Солнечной системе. 
Положения этих трёх тел в момент времени t обозначим как q_i(t), где i = 0 соответствует Солнцу, i = 1 - Земле, i = 2 - Луне. 
Массы тел обозначим как m_i. Также рассматриваем импульсы этих тел.  
**_p_i(t) = m_i * g_i'(t)_**  (*)
По 2 закону Ньютона:  
![img_1.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/img_1.png) (**)

Взяв (∗) и (∗∗) вместе и, выразив правые части через (∗∗), используя закон всемирного тяготения Ньютона, 
получим систему из шести дифференциальных уравнений
для функций q_i и p_i, к которым применим численные методы интегрирования.

---

# 3. Результаты
### Модель Солнце-Земля-Луна  
![img_2.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/the%20Sun-Earth-Moon%20model.png)
## Метод Рунге-Кутты 4 порядка
### Расстояние между Солнцем и Луной(шаг 0.1)
![img_3.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/RK4_0.1.png)
### Расстояние между Солнцем и Луной(шаг 0.01)
![img_4.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/RK4_0.01.png)

## Метод Дормана-Принса
### Расстояние между Солнцем и Луной(шаг 1)
![img_6.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/DP_1.png)
### Расстояние между Солнцем и Луной(шаг 0.1)
![img_5.png](https://github.com/Natalya00/Building-orbits-of-celestial-bodies/blob/main/graphics/DP_0.1.png)
