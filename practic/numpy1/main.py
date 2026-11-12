import time
import numpy as np


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        
        if t1 - t0 < best:
            best = t1 - t0
            
    return result, best


def task1():
    print("")
    print("Задача 1. Сумма массива")
    print("")

    def loop_sum(a):
        s = 0.0
        for x in a:
            s += x
        return s

    def numpy_sum(a):
        return a.sum()

    rng = np.random.default_rng(42)
    
    for n in (10**5, 10**6, 10**7):
        a = rng.normal(size=n)
        r_loop, t_loop = measure(loop_sum, a)
        r_np, t_np = measure(numpy_sum, a)

        print(f"n = {n:>9}: цикл = {t_loop*1000:9.2f} мс, "
              f"NumPy = {t_np*1000:8.2f} мс, ускорение = {t_loop/t_np:6.1f}x, "
              f"совпадает = {np.isclose(r_loop, r_np)}")
    print("")


def task2():
    print("")
    print("Задача 2. Среднее и стандартное отклонение")
    print("")

    def loop_mean_std(a):
        n = len(a)
        s = 0.0
        for x in a:
            s += x
            
        mean = s / n
        sq = 0.0
        
        for x in a:
            d = x - mean
            sq += d * d
            
        std = (sq / n) ** 0.5
        
        return mean, std

    def numpy_mean_std(a):
        return a.mean(), a.std()

    rng = np.random.default_rng(42)
    a = rng.normal(size=10**6)

    (m_l, s_l), t_loop = measure(loop_mean_std, a)
    (m_n, s_n), t_np = measure(numpy_mean_std, a)

    print(f"цикл:  mean = {m_l:.15f}, std = {s_l:.15f}, время = {t_loop*1000:.2f} мс")
    print(f"NumPy: mean = {m_n:.15f}, std = {s_n:.15f}, время = {t_np*1000:.2f} мс")
    print(f"ускорение: {t_loop/t_np:.1f}x")
    print(f"|mean diff| = {abs(m_l - m_n):.2e}")
    print(f"|std  diff| = {abs(s_l - s_n):.2e}")
    print(f"совпадает до 1e-12: {abs(m_l - m_n) < 1e-12 and abs(s_l - s_n) < 1e-12}")
    print("")


def task3():
    print("")
    print("Задача 3. y = sin(x) + x^2")
    print("")

    def loop_transform(a):
        out = np.empty(len(a))
        
        for i in range(len(a)):
            x = a[i]
            out[i] = np.sin(x) + x * x
            
        return out

    def numpy_transform(a):
        return np.sin(a) + a ** 2

    rng = np.random.default_rng(42)
    a = rng.normal(size=10**6)

    r_loop, t_loop = measure(loop_transform, a)
    r_np, t_np = measure(numpy_transform, a)

    print(f"цикл:  {t_loop*1000:9.2f} мс")
    print(f"NumPy: {t_np*1000:9.2f} мс")
    print(f"ускорение: {t_loop/t_np:.1f}x")
    print(f"совпадает: {np.allclose(r_loop, r_np)}")
    print()


def task4():
    print("")
    print("Задача 4. Условная замена (clipping)")
    print("")

    def loop_clip(a):
        out = np.empty(len(a))
        
        for i in range(len(a)):
            x = a[i]
            
            if x < 0:
                out[i] = 0
                
            elif x > 1:
                out[i] = 1
                
            else:
                out[i] = x
                
        return out

    def numpy_clip(a):
        return np.clip(a, 0, 1)

    rng = np.random.default_rng(42)
    a = rng.normal(size=10**6)

    r_loop, t_loop = measure(loop_clip, a)
    r_np, t_np = measure(numpy_clip, a)

    changed = int(np.sum((a < 0) | (a > 1)))

    print(f"цикл:  {t_loop*1000:9.2f} мс")
    print(f"NumPy: {t_np*1000:9.2f} мс")
    print(f"ускорение: {t_loop/t_np:.1f}x")
    print(f"совпадает: {np.allclose(r_loop, r_np)}")
    print(f"элементов изменено: {changed} из {len(a)}")
    print()

def main():
    task1()
    task2()
    task3()
    task4()

main()
