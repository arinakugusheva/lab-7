
"""
Лабораторная работа № 7. Вариант 13.
Дисциплина: Теория алгоритмов.
Выполнила: Кугушева Арина Вячеславовна, группа БИ 2-1.

Тема: рекурсивные алгоритмы и динамическое программирование (ДП) в задачах
комбинаторной оптимизации.

ИСХОДНЫЕ ДАННЫЕ ВАРИАНТА 13 (табл. 1 задания)
    задача А (три реализации) .. число способов подняться по лестнице из n
                                 ступеней шагами по 1, 2 или 3 ступени;
                                 параметр n = 10 ... 27;
    задача Б (ДП) .............. редакционное расстояние (Левенштейна) между
                                 двумя строками длиной 300 символов;
    дополнительное задание Г6 .. сравнить собственную мемоизацию с декоратором
                                 functools.lru_cache и объяснить разницу во времени.

ЗАДАЧА А. Рекуррентное соотношение:
    f(0) = f(1) = 1,  f(2) = 2,  f(n) = f(n-1) + f(n-2) + f(n-3)  при n >= 3.
    (На последней ступени оказываются, сделав шаг в 1, 2 или 3 ступени с
    ступеней n-1, n-2, n-3 соответственно.) Три реализации: наивная рекурсия,
    рекурсия с мемоизацией, итеративное решение; все считают число вызовов.

ЗАДАЧА Б. Редакционное расстояние D[i][j] между первыми i символами строки A
и первыми j символами строки B (операции: вставка, удаление, замена, цена 1):
    D[i][0] = i,  D[0][j] = j,
    D[i][j] = D[i-1][j-1],                                   если A[i] = B[j],
    D[i][j] = 1 + min(D[i-1][j], D[i][j-1], D[i-1][j-1]),    иначе.
Программа не только считает расстояние, но и ВОССТАНАВЛИВАЕТ оптимальную
последовательность операций обратным проходом по таблице.

ЧТО ДЕЛАЕТ ПРОГРАММА (по порядку)
    0. Самопроверка: ДП сверяется с полным перебором на 200 парах малых строк,
       восстановленная последовательность операций применяется к строке A и
       должна дать строку B.
    1. Дерево рекурсии для f(5) и таблица ДП для малого примера.
    2. Задача А: число вызовов и время трёх реализаций при n = 10 ... 27,
       оценка основания экспоненты, поиск n, при котором наивная рекурсия
       перестаёт укладываться в 10 секунд.
    3. Задача Б: оптимальное решение для строк длиной 300 (с расшифровкой);
       наивная рекурсия, мемоизация и таблица на малых длинах; рост времени ДП
       с длиной строк (должен быть квадратичным).
    4. Г6: собственная мемоизация и lru_cache - время и число вызовов.

ЗАПУСК
    python lr7_variant13.py             (полный прогон, около 1-1,5 минуты)
    python lr7_variant13.py --json F    (дополнительно сохранить результаты в F)
    python lr7_variant13.py --show      (дополнительно показать графики)

Требуется: Python 3.10+, библиотека matplotlib (pip install matplotlib).
Результат: таблицы в консоли и графики lr7_plot_stairs.png,
lr7_plot_edit.png, lr7_plot_cache.png в текущей папке.
"""

import functools
import json
import math
import platform
import random
import statistics
import sys
import time

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")          # режим без окна: графики только сохраняются в файлы
import matplotlib.pyplot as plt

sys.setrecursionlimit(20_000)      # мемоизированная рекурсия для строк длиной 300 уходит на ~600 уровней

# --------------------------------------------------------------------------
# 1. ПАРАМЕТРЫ ЭКСПЕРИМЕНТА
# --------------------------------------------------------------------------
STAIRS_N = list(range(10, 28))      # параметр n задачи А: 10 ... 27 (вариант 13)
LIMIT_SECONDS = 10.0                # «разумное время» для наивной рекурсии (ориентир из задания)
LIMIT_SEARCH_START = 28             # с этого n ищем предел наивной рекурсии
LIMIT_SEARCH_MAX = 36               # выше не идём в любом случае
STRING_LEN = 300                    # длина строк задачи Б (вариант 13)
MUTATIONS = 30                      # число случайных правок, превращающих A в B
ALPHABET = "абвгдежзиклмнопрстуфхцчшэюя"   # алфавит для генерации строк
NAIVE_EDIT_LENGTHS = list(range(2, 9))      # длины строк для наивного перебора в задаче Б
EDIT_LENGTHS = [50, 100, 150, 200, 300, 400]   # длины для замеров роста времени ДП
CACHE_BIG_N = [100, 300, 900]       # большие n для сравнения мемоизаций в задаче А
CACHE_EDIT_LENGTHS = [100, 200, 300]   # длины для сравнения мемоизаций в задаче Б
SEED = 13                           # начальное значение генератора случайных чисел


# --------------------------------------------------------------------------
# 2. ЗАДАЧА А: ЧИСЛО СПОСОБОВ ПОДНЯТЬСЯ ПО ЛЕСТНИЦЕ (ШАГИ 1, 2, 3)
# --------------------------------------------------------------------------
def stairs_naive(n):
    """
    НАИВНАЯ рекурсия по определению f(n) = f(n-1) + f(n-2) + f(n-3).
    Одни и те же подзадачи решаются многократно, число вызовов растёт
    экспоненциально: T(n) = 1 + T(n-1) + T(n-2) + T(n-3), T(0) = T(1) = T(2) = 1,
    то есть Θ(c^n), где c ≈ 1,8393 - корень уравнения x^3 = x^2 + x + 1.
    Возвращает пару (значение f(n), число вызовов функции).
    """
    calls = [0]                                     # счётчик вызовов (список - чтобы менять из вложенной функции)

    def f(k):
        calls[0] += 1                               # каждый вызов, включая базовые, учитывается
        if k <= 1:                                  # базовый случай: f(0) = f(1) = 1
            return 1
        if k == 2:                                  # базовый случай: f(2) = 2
            return 2
        return f(k - 1) + f(k - 2) + f(k - 3)       # шаг рекурсии

    return f(n), calls[0]


def stairs_memo(n):
    """
    РЕКУРСИЯ С МЕМОИЗАЦИЕЙ (нисходящее ДП): результат каждой подзадачи один раз
    записывается в словарь cache и затем берётся оттуда.
    Нетривиальных подзадач (k = 3 ... n) всего n - 2, каждая вычисляется один
    раз и порождает три вызова, поэтому число вызовов равно 1 + 3(n - 2) = 3n - 5
    при n >= 3, то есть Θ(n); память Θ(n) на кэш и стек рекурсии глубиной n.
    Возвращает пару (значение, число вызовов).
    """
    cache = {}                                      # кэш: номер ступени -> число способов
    calls = [0]

    def f(k):
        calls[0] += 1
        if k <= 1:
            return 1
        if k == 2:
            return 2
        if k in cache:                              # подзадача уже решена - берём из кэша
            return cache[k]
        cache[k] = f(k - 1) + f(k - 2) + f(k - 3)   # решаем один раз и запоминаем
        return cache[k]

    return f(n), calls[0]


def stairs_iter(n):
    """
    ИТЕРАТИВНОЕ решение (восходящее ДП): значения f(0), f(1), ... вычисляются по
    порядку, хранятся только три последних. Θ(n) времени, O(1) памяти,
    рекурсии нет - нет и ограничения на глубину стека.
    Возвращает пару (значение, число «вызовов»). Вызов тут один (функции
    не вызывают друг друга), а число шагов цикла - n - 2, поэтому в счётчик
    записывается именно число шагов цикла (+1 за сам вызов), чтобы сравнивать
    объём работы с другими версиями.
    """
    if n <= 1:
        return 1, 1
    if n == 2:
        return 2, 1
    a, b, c = 1, 1, 2                               # f(0), f(1), f(2)
    steps = 1                                       # один вызов самой функции
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c                   # сдвиг окна из трёх значений
        steps += 1
    return c, steps


def stairs_calls_formula(n):
    """Число вызовов наивной рекурсии по формуле T(n) = 1 + T(n-1) + T(n-2) + T(n-3)."""
    t = [1, 1, 1]                                   # T(0), T(1), T(2)
    for k in range(3, n + 1):
        t.append(1 + t[k - 1] + t[k - 2] + t[k - 3])
    return t[n]


def recursion_tree_stairs(n=5):
    """
    Дерево рекурсии наивной версии для f(n) в формате Mermaid (для отчёта).
    Повторяющиеся подзадачи помечаются номером вхождения: F(3) ①, F(3) ② ...
    Возвращает строку с кодом диаграммы и словарь «подзадача -> число вхождений».
    """
    lines, counts, seen = ["flowchart TD"], {}, {}
    uid = [0]

    def build(k):
        uid[0] += 1
        node = f"N{uid[0]}"
        counts[k] = counts.get(k, 0) + 1
        seen[k] = seen.get(k, 0) + 1
        label = f"F({k})"
        lines.append(f'    {node}["{label}"]')       # подпись уточняется после обхода
        if k > 2:
            for d in (1, 2, 3):                     # три дочерних вызова
                child = build(k - d)
                lines.append(f"    {node} --> {child}")
        return node

    build(n)
    # проставляем номера вхождений для подзадач, которые повторяются
    order, out = {}, []
    for line in lines:
        if line.strip().startswith("N") and '["F(' in line:
            k = int(line.split("F(")[1].split(")")[0])
            if counts[k] > 1:
                order[k] = order.get(k, 0) + 1
                mark = "①②③④⑤⑥⑦⑧⑨"[order[k] - 1]
                line = line.replace(f'F({k})"', f'F({k}) {mark}"')
        out.append(line)
    return "\n".join(out), counts


# --------------------------------------------------------------------------
# 3. ЗАДАЧА Б: РЕДАКЦИОННОЕ РАССТОЯНИЕ
# --------------------------------------------------------------------------
def edit_naive(a, b):
    """
    ПОЛНЫЙ ПЕРЕБОР (наивная рекурсия по определению). Годится только для очень
    коротких строк: число вызовов растёт примерно как (3 + 2·√2)^n ≈ 5,83^n.
    Возвращает пару (расстояние, число вызовов).
    """
    calls = [0]

    def d(i, j):                                    # расстояние между a[:i] и b[:j]
        calls[0] += 1
        if i == 0:                                  # A пуста: нужно вставить j символов
            return j
        if j == 0:                                  # B пуста: нужно удалить i символов
            return i
        if a[i - 1] == b[j - 1]:                    # последние символы совпали - ничего не делаем
            return d(i - 1, j - 1)
        return 1 + min(d(i - 1, j),                 # удаление a[i-1]
                       d(i, j - 1),                 # вставка b[j-1]
                       d(i - 1, j - 1))             # замена a[i-1] на b[j-1]

    return d(len(a), len(b)), calls[0]


def edit_memo(a, b):
    """
    Рекурсия с МЕМОИЗАЦИЕЙ: словарь cache хранит D[i][j]. Различных подзадач
    (m + 1)(n + 1), каждая считается один раз: Θ(mn) времени и памяти.
    Возвращает пару (расстояние, число вызовов).
    """
    cache = {}
    calls = [0]

    def d(i, j):
        calls[0] += 1
        if i == 0:
            return j
        if j == 0:
            return i
        key = (i, j)
        if key in cache:                            # подзадача уже решена
            return cache[key]
        if a[i - 1] == b[j - 1]:
            res = d(i - 1, j - 1)
        else:
            res = 1 + min(d(i - 1, j), d(i, j - 1), d(i - 1, j - 1))
        cache[key] = res
        return res

    return d(len(a), len(b)), calls[0]


def edit_table(a, b):
    """
    ВОСХОДЯЩЕЕ ДП: заполняется вся таблица D размером (m + 1) x (n + 1) строка
    за строкой слева направо; рекурсии нет. Θ(mn) времени и памяти.
    Возвращает таблицу D (список списков); расстояние лежит в D[m][n].
    """
    m, n = len(a), len(b)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):                          # первый столбец: D[i][0] = i
        D[i][0] = i
    for j in range(n + 1):                          # первая строка: D[0][j] = j
        D[0][j] = j
    for i in range(1, m + 1):
        row, prev = D[i], D[i - 1]                  # локальные ссылки на строки - чуть быстрее
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                row[j] = prev[j - 1]                # символы совпали
            else:
                row[j] = 1 + min(prev[j],           # удаление
                                 row[j - 1],        # вставка
                                 prev[j - 1])       # замена
    return D


def edit_two_rows(a, b):
    """
    ДП с ОПТИМИЗАЦИЕЙ ПАМЯТИ: в каждый момент нужны только предыдущая и текущая
    строки таблицы, поэтому память O(n) вместо O(mn). Платой за это служит
    невозможность восстановить последовательность операций по таблице.
    Возвращает расстояние.
    """
    n = len(b)
    prev = list(range(n + 1))                       # строка D[0][*]
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * n                         # D[i][0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n]


def edit_script(a, b, D):
    """
    ВОССТАНОВЛЕНИЕ решения обратным проходом по таблице D: от клетки (m, n) к
    (0, 0). В каждой клетке выясняем, из какой соседней клетки получено её
    значение. Приоритет при равенстве: совпадение/замена, затем удаление,
    затем вставка (так результат воспроизводим).

    Возвращает список операций от начала строки к концу. Каждая операция -
    кортеж (название, i, j, символ_A, символ_B), где i и j - позиции (с 1)
    в строках A и B; названия: "совпадение", "замена", "удаление", "вставка".
    """
    ops = []
    i, j = len(a), len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            ops.append(("совпадение", i, j, a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            ops.append(("замена", i, j, a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("удаление", i, j, a[i - 1], ""))
            i -= 1
        else:                                       # остаётся вставка: D[i][j] = D[i][j-1] + 1
            ops.append(("вставка", i, j, "", b[j - 1]))
            j -= 1
    ops.reverse()                                   # шли с конца - разворачиваем
    return ops


def apply_script(a, ops):
    """Применяет последовательность операций к строке A и возвращает полученную строку."""
    result = []
    for name, i, j, ca, cb in ops:
        if name in ("совпадение", "замена", "вставка"):
            result.append(cb)                       # символ B попадает в результат
        # при удалении символ A просто пропускается
    return "".join(result)


def random_text(rng, n):
    """Случайная строка длиной n над русским алфавитом."""
    return "".join(rng.choice(ALPHABET) for _ in range(n))


def mutate(rng, a, k):
    """
    Строка B из A: выполняется k случайных правок, причём длина сохраняется
    (поэтому обе строки имеют одинаковую длину, 300). Правка - это либо замена
    случайного символа на случайный, либо «сдвиг»: удаление символа в одной
    позиции и вставка случайного символа в другой.
    """
    chars = list(a)
    for _ in range(k):
        if rng.random() < 0.5:                      # замена символа
            chars[rng.randrange(len(chars))] = rng.choice(ALPHABET)
        else:                                       # удаление + вставка в другом месте
            del chars[rng.randrange(len(chars))]
            chars.insert(rng.randrange(len(chars) + 1), rng.choice(ALPHABET))
    return "".join(chars)


# --------------------------------------------------------------------------
# 4. ГЕНЕРИЧЕСКИЕ МЕМОИЗАЦИИ ДЛЯ СРАВНЕНИЯ (Г6)
# --------------------------------------------------------------------------
def my_memoize(func):
    """
    СОБСТВЕННЫЙ декоратор мемоизации (чистый Python): словарь cache и обёртка.
    Ключом служит кортеж аргументов. Статистика calls / misses нужна для отчёта.
    """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        wrapper.calls += 1                          # все обращения к функции
        if args in cache:                           # попадание в кэш
            return cache[args]
        wrapper.misses += 1                         # реальное вычисление
        result = func(*args)
        cache[args] = result
        return result

    wrapper.calls = 0
    wrapper.misses = 0
    return wrapper


def make_stairs_decorated(kind):
    """
    Создаёт НОВУЮ (с пустым кэшем) мемоизированную функцию f(n) задачи А.
    kind: "own" - собственный декоратор my_memoize, "lru" - functools.lru_cache.
    """
    def f(k):
        if k <= 1:
            return 1
        if k == 2:
            return 2
        return g(k - 1) + g(k - 2) + g(k - 3)       # рекурсия идёт через мемоизированную g

    g = my_memoize(f) if kind == "own" else functools.lru_cache(maxsize=None)(f)
    return g


def make_edit_decorated(kind, a, b):
    """То же для редакционного расстояния: возвращает мемоизированную d(i, j)."""
    def d(i, j):
        if i == 0:
            return j
        if j == 0:
            return i
        if a[i - 1] == b[j - 1]:
            return g(i - 1, j - 1)
        return 1 + min(g(i - 1, j), g(i, j - 1), g(i - 1, j - 1))

    g = my_memoize(d) if kind == "own" else functools.lru_cache(maxsize=None)(d)
    return g


def cache_calls(g, kind):
    """(всего обращений, реальных вычислений) для мемоизированной функции g."""
    if kind == "own":
        return g.calls, g.misses
    info = g.cache_info()
    return info.hits + info.misses, info.misses


# --------------------------------------------------------------------------
# 5. ИЗМЕРЕНИЯ
# --------------------------------------------------------------------------
def timed(func):
    """Время выполнения func() в секундах (и её результат)."""
    t0 = time.perf_counter()
    res = func()
    return time.perf_counter() - t0, res


def avg_time(func, min_total=0.05, max_reps=20_000):
    """
    Среднее время одного вызова func() в секундах для БЫСТРЫХ функций: вызовы
    повторяются, пока суммарное время не превысит min_total. Возвращает медиану
    по пяти таким сериям.
    """
    t, _ = timed(func)                              # пробный вызов для оценки времени
    reps = max(1, min(max_reps, int(min_total / max(t, 1e-7))))
    series = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(reps):
            func()
        series.append((time.perf_counter() - t0) / reps)
    return statistics.median(series)


def median_run(maker, reps):
    """
    Медиана времени для функций, которым нужен СВЕЖИЙ кэш: maker() создаёт
    функцию заново (создание в замер не входит), затем замеряется один вызов.
    maker должна возвращать пару (функция без аргументов, объект_для_счётчика).
    """
    times = []
    for _ in range(reps):
        call, _ = maker()
        t, _ = timed(call)
        times.append(t)
    return statistics.median(times)


def experiment_stairs():
    """Задача А: число вызовов и время трёх реализаций при n = 10 ... 27."""
    rows = []
    for n in STAIRS_N:
        v1, c1 = stairs_naive(n)
        v2, c2 = stairs_memo(n)
        v3, c3 = stairs_iter(n)
        assert v1 == v2 == v3, "реализации дали разные значения"
        assert c1 == stairs_calls_formula(n), "число вызовов не совпало с формулой T(n)"
        assert c2 == 3 * n - 5, "число вызовов мемоизации не совпало с 3n - 5"
        if n <= 22:                                 # быстрые случаи измеряем усреднением
            t1 = avg_time(lambda: stairs_naive(n), 0.2, 50)
        else:                                       # крупные - единичным замером (медиана из 3 для n <= 25)
            reps = 3 if n <= 25 else 1
            t1 = statistics.median(timed(lambda: stairs_naive(n))[0] for _ in range(reps))
        rows.append({
            "n": n, "value": v1,
            "calls_naive": c1, "calls_memo": c2, "calls_iter": c3,
            "t_naive": t1,
            "t_memo": avg_time(lambda: stairs_memo(n)),
            "t_iter": avg_time(lambda: stairs_iter(n)),
        })
    return rows


def experiment_limit(last_n, last_time):
    """
    Поиск n, при котором наивная рекурсия перестаёт укладываться в LIMIT_SECONDS.
    Запускаем наивную версию для n = LIMIT_SEARCH_START, +1, ... до первого
    превышения порога (но не выше LIMIT_SEARCH_MAX).
    """
    out = []
    for n in range(LIMIT_SEARCH_START, LIMIT_SEARCH_MAX + 1):
        t, (v, c) = timed(lambda: stairs_naive(n))
        out.append({"n": n, "t": t, "calls": c, "value": v})
        print(f"    n = {n}: {t:.2f} с, {c} вызовов", flush=True)
        if t > LIMIT_SECONDS:
            break
    return out


def fit_base(rows):
    """Основание экспоненты: среднее T(n+1)/T(n) и оценка по наклону log T от n."""
    ratios = [rows[i + 1]["calls_naive"] / rows[i]["calls_naive"] for i in range(len(rows) - 1)]
    xs = [r["n"] for r in rows]
    ys = [math.log(r["calls_naive"]) for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return {"last_ratio": ratios[-1], "mean_ratio": statistics.mean(ratios),
            "slope_base": math.exp(slope)}


def experiment_edit_main():
    """
    Основной случай задачи Б: две строки длиной 300 (B получена из A правками).
    Возвращает расстояние, расшифровку и сопоставление реализаций.
    """
    rng = random.Random(SEED)
    a = random_text(rng, STRING_LEN)
    b = mutate(rng, a, MUTATIONS)
    D = edit_table(a, b)
    dist = D[len(a)][len(b)]
    ops = edit_script(a, b, D)
    assert apply_script(a, ops) == b, "восстановленная последовательность не даёт строку B"
    cost = sum(1 for o in ops if o[0] != "совпадение")
    assert cost == dist, "цена последовательности не равна расстоянию"
    assert edit_two_rows(a, b) == dist
    assert edit_memo(a, b)[0] == dist
    # контраст: две независимые случайные строки той же длины
    c = random_text(rng, STRING_LEN)
    dist_random = edit_two_rows(a, c)
    counts = {name: sum(1 for o in ops if o[0] == name)
              for name in ("совпадение", "замена", "удаление", "вставка")}
    return {"a": a, "b": b, "distance": dist, "ops": ops, "counts": counts,
            "distance_independent": dist_random, "cells": (len(a) + 1) * (len(b) + 1)}


def experiment_edit_naive():
    """
    Малые длины: наивный перебор, мемоизация и таблица. Берутся строки БЕЗ общих
    символов (A над первой половиной алфавита, B - над второй): это худший
    случай для перебора, число вызовов от строки к строке не «прыгает».
    Для каждой длины - число вызовов и время.
    """
    rng = random.Random(SEED + 1)
    half = len(ALPHABET) // 2
    rows = []
    for L in NAIVE_EDIT_LENGTHS:
        a = "".join(rng.choice(ALPHABET[:half]) for _ in range(L))
        b = "".join(rng.choice(ALPHABET[half:]) for _ in range(L))
        dn, cn = edit_naive(a, b)
        dm, cm = edit_memo(a, b)
        dt = edit_table(a, b)[L][L]
        assert dn == dm == dt
        rows.append({
            "L": L, "distance": dn, "calls_naive": cn, "calls_memo": cm, "cells": (L + 1) ** 2,
            "t_naive": avg_time(lambda: edit_naive(a, b), 0.1, 200),
            "t_memo": avg_time(lambda: edit_memo(a, b)),
            "t_table": avg_time(lambda: edit_table(a, b)),
        })
    return rows


def experiment_edit_scaling():
    """Рост времени таблицы и двухстрочной версии с длиной строк (ожидается Θ(L²))."""
    rng = random.Random(SEED + 2)
    rows = []
    for L in EDIT_LENGTHS:
        a, b = random_text(rng, L), random_text(rng, L)
        rows.append({
            "L": L, "cells": (L + 1) ** 2,
            "t_table": avg_time(lambda: edit_table(a, b), 0.2, 200),
            "t_two_rows": avg_time(lambda: edit_two_rows(a, b), 0.2, 200),
        })
    return rows


def experiment_cache():
    """
    Г6: собственная мемоизация (декоратор my_memoize), lru_cache и явный словарь
    (функции stairs_memo / edit_memo). Время вычисления с ПУСТЫМ кэшем и число
    обращений / реальных вычислений. Для задачи А - при n из варианта (27) и при
    больших n; для задачи Б - при длинах CACHE_EDIT_LENGTHS.
    """
    out = {"stairs": [], "edit": []}
    for n in [27] + CACHE_BIG_N:
        row = {"n": n}
        for kind in ("own", "lru"):
            def maker(kind=kind):
                g = make_stairs_decorated(kind)
                return (lambda: g(n)), g
            row["t_" + kind] = median_run(maker, 200 if n < 100 else 100)
            g = make_stairs_decorated(kind)
            g(n)
            row["calls_" + kind], row["misses_" + kind] = cache_calls(g, kind)
        row["t_dict"] = avg_time(lambda: stairs_memo(n))      # явный словарь внутри функции
        row["calls_dict"] = stairs_memo(n)[1]
        # проверка совпадения значений
        assert make_stairs_decorated("own")(n) == make_stairs_decorated("lru")(n) == stairs_memo(n)[0]
        out["stairs"].append(row)
    rng = random.Random(SEED + 3)
    for L in CACHE_EDIT_LENGTHS:
        a, b = random_text(rng, L), random_text(rng, L)
        row = {"L": L}
        for kind in ("own", "lru"):
            def maker(kind=kind):
                g = make_edit_decorated(kind, a, b)
                return (lambda: g(len(a), len(b))), g
            row["t_" + kind] = median_run(maker, 7)
            g = make_edit_decorated(kind, a, b)
            d = g(len(a), len(b))
            row["calls_" + kind], row["misses_" + kind] = cache_calls(g, kind)
            row["distance"] = d
        row["t_dict"] = avg_time(lambda: edit_memo(a, b), 0.2, 50)
        row["calls_dict"] = edit_memo(a, b)[1]
        row["t_table"] = avg_time(lambda: edit_table(a, b), 0.2, 50)
        assert row["distance"] == edit_table(a, b)[L][L]
        out["edit"].append(row)
    return out


# --------------------------------------------------------------------------
# 6. САМОПРОВЕРКА И МАЛЫЕ ПРИМЕРЫ
# --------------------------------------------------------------------------
def self_test():
    """
    Проверки: (1) три реализации задачи А совпадают с прямым перебором путей;
    (2) ДП для редакционного расстояния совпадает с полным перебором на 200
    парах малых строк; (3) восстановленная последовательность операций
    превращает A в B и стоит ровно D[m][n]. При ошибке - AssertionError.
    """
    # (1) прямое построение всех способов (последовательностей шагов 1, 2, 3)
    def brute_stairs(n):
        if n == 0:
            return 1
        return sum(brute_stairs(n - s) for s in (1, 2, 3) if n - s >= 0)
    for n in range(0, 16):
        assert stairs_naive(n)[0] == stairs_memo(n)[0] == stairs_iter(n)[0] == brute_stairs(n)
    # (2) и (3) редакционное расстояние
    rng = random.Random(0)
    for _ in range(200):
        la, lb = rng.randint(0, 7), rng.randint(0, 7)
        a = "".join(rng.choice("абв") for _ in range(la))      # малый алфавит - много совпадений
        b = "".join(rng.choice("абв") for _ in range(lb))
        D = edit_table(a, b)
        dist = D[la][lb]
        assert edit_naive(a, b)[0] == dist, "ДП не совпало с полным перебором"
        assert edit_memo(a, b)[0] == dist and edit_two_rows(a, b) == dist
        ops = edit_script(a, b, D)
        assert apply_script(a, ops) == b
        assert sum(1 for o in ops if o[0] != "совпадение") == dist
    print("Самопроверка пройдена (задача А: 3 реализации = перебор; задача Б: ДП = полный перебор на 200 парах)")


def demo_small():
    """Малые примеры: дерево рекурсии f(5), таблица ДП и расшифровка для двух слов."""
    tree, counts = recursion_tree_stairs(5)
    print("\n--- Дерево рекурсии наивной версии для f(5) (Mermaid) ---")
    print(tree)
    print("Число вхождений подзадач:", ", ".join(f"F({k}) x{v}" for k, v in sorted(counts.items(), reverse=True)),
          f"; всего вызовов {sum(counts.values())}")

    a, b = "СОБАКА", "ЛОПАТА"
    D = edit_table(a, b)
    print(f"\n--- Таблица ДП для слов «{a}» и «{b}» ---")
    print("      " + "   ".join(["·"] + list(b)))
    for i in range(len(a) + 1):
        label = "·" if i == 0 else a[i - 1]
        print(f"  {label}  " + "  ".join(f"{D[i][j]:>2}" for j in range(len(b) + 1)))
    ops = edit_script(a, b, D)
    print(f"Расстояние = {D[len(a)][len(b)]}. Расшифровка:")
    for name, i, j, ca, cb in ops:
        print(f"  {name:<11} {('A[' + str(i) + ']=' + ca) if ca else '':<9} {('B[' + str(j) + ']=' + cb) if cb else ''}")
    print("Результат применения:", apply_script(a, ops))


# --------------------------------------------------------------------------
# 7. ГРАФИКИ
# --------------------------------------------------------------------------
def make_plots(res):
    """Строит и сохраняет три рисунка (lr7_plot_stairs / _edit / _cache)."""
    rows = res["stairs"]
    ns = [r["n"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(ns, [r["calls_naive"] for r in rows], "ro-", label="наивная рекурсия")
    ax.plot(ns, [r["calls_memo"] for r in rows], "bs-", label="мемоизация")
    ax.plot(ns, [r["calls_iter"] for r in rows], "g^-", label="итеративная (шаги цикла)")
    base = res["fit"]["slope_base"]
    c0 = rows[0]["calls_naive"]
    ax.plot(ns, [c0 * base ** (n - ns[0]) for n in ns], "k--", lw=1, label=f"c^n, c = {base:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("число вызовов (лог. шкала)")
    ax.set_title("Задача А: число вызовов")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(ns, [r["t_naive"] for r in rows], "ro-", label="наивная рекурсия")
    ax.plot(ns, [r["t_memo"] for r in rows], "bs-", label="мемоизация")
    ax.plot(ns, [r["t_iter"] for r in rows], "g^-", label="итеративная")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("время, с (лог. шкала)")
    ax.set_title("Задача А: время работы")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("lr7_plot_stairs.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    nv = res["edit_naive"]
    ax.plot([r["L"] for r in nv], [r["calls_naive"] for r in nv], "ro-", label="полный перебор")
    ax.plot([r["L"] for r in nv], [r["calls_memo"] for r in nv], "bs-", label="мемоизация")
    ax.plot([r["L"] for r in nv], [r["cells"] for r in nv], "g^-", label="клеток таблицы (L+1)²")
    ax.set_yscale("log")
    ax.set_xlabel("длина строк L")
    ax.set_ylabel("число вызовов (лог. шкала)")
    ax.set_title("Задача Б: малые длины")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    sc = res["edit_scaling"]
    ax.plot([r["L"] for r in sc], [r["t_table"] * 1e3 for r in sc], "bo-", label="таблица")
    ax.plot([r["L"] for r in sc], [r["t_two_rows"] * 1e3 for r in sc], "g^-", label="две строки")
    l0, t0 = sc[0]["L"], sc[0]["t_table"] * 1e3
    ax.plot([r["L"] for r in sc], [t0 * (r["L"] / l0) ** 2 for r in sc], "k--", lw=1, label="∝ L²")
    ax.set_xlabel("длина строк L")
    ax.set_ylabel("время, мс")
    ax.set_title("Задача Б: рост времени ДП")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("lr7_plot_edit.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cs, ce = res["cache"]["stairs"], res["cache"]["edit"]
    for ax, data, key, title, unit in ((axes[0], cs, "n", "Задача А (лестница)", 1e6),
                                       (axes[1], ce, "L", "Задача Б (редакционное расстояние)", 1e3)):
        xs = range(len(data))
        w = 0.27
        ax.bar([x - w for x in xs], [d["t_own"] * unit for d in data], w, label="собственный декоратор")
        ax.bar(list(xs), [d["t_lru"] * unit for d in data], w, label="functools.lru_cache")
        ax.bar([x + w for x in xs], [d["t_dict"] * unit for d in data], w, label="явный словарь в функции")
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{key} = {d[key]}" for d in data])
        ax.set_ylabel("время, мкс" if unit == 1e6 else "время, мс")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("lr7_plot_cache.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 8. ВЫВОД РЕЗУЛЬТАТОВ
# --------------------------------------------------------------------------
def system_info():
    """Строка с характеристиками системы (вставляется в отчёт)."""
    cpu = platform.processor() or platform.machine()
    try:                                            # на Linux модель процессора лежит в /proc/cpuinfo
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"ОС: {platform.system()} {platform.release()}; процессор: {cpu}; "
            f"Python {platform.python_version()}")


def print_report(res):
    """Печатает таблицы результатов в консоль."""
    print("\nТаблица 1. Задача А: значение f(n), число вызовов и время")
    print(f"  {'n':>3}{'f(n)':>12}{'вызовов (наив.)':>17}{'вызовов (мемо)':>16}{'шагов (итер.)':>15}"
          f"{'t наив., с':>13}{'t мемо, мкс':>13}{'t итер., мкс':>14}")
    for r in res["stairs"]:
        print(f"  {r['n']:>3}{r['value']:>12}{r['calls_naive']:>17}{r['calls_memo']:>16}{r['calls_iter']:>15}"
              f"{r['t_naive']:>13.5f}{r['t_memo'] * 1e6:>13.2f}{r['t_iter'] * 1e6:>14.2f}")
    f = res["fit"]
    print(f"  Основание экспоненты: T(n+1)/T(n) при n = 27: {f['last_ratio']:.4f}; среднее: "
          f"{f['mean_ratio']:.4f}; по наклону log T: {f['slope_base']:.4f}; теория (корень x³ = x² + x + 1): 1.8393")

    print("\nТаблица 2. Предел наивной рекурсии (порог 10 с)")
    for r in res["limit"]:
        print(f"  n = {r['n']}: {r['t']:.2f} с, {r['calls']} вызовов")

    e = res["edit_main"]
    print(f"\nТаблица 3. Задача Б: строки длиной {STRING_LEN}, A -> B ({MUTATIONS} случайных правок)")
    print(f"  Редакционное расстояние = {e['distance']} (для двух независимых случайных строк: {e['distance_independent']})")
    print("  Состав: " + ", ".join(f"{k} - {v}" for k, v in e["counts"].items()))
    print("  Расшифровка (без совпадений): №, операция, позиция в A, позиция в B, символы")
    k = 0
    for name, i, j, ca, cb in e["ops"]:
        if name != "совпадение":
            k += 1
            print(f"    {k:>3}. {name:<8} A[{i}]={ca or '-'}  B[{j}]={cb or '-'}")

    print("\nТаблица 4. Задача Б, малые длины: число вызовов и время")
    print(f"  {'L':>3}{'D':>4}{'вызовов (перебор)':>19}{'вызовов (мемо)':>16}{'клеток':>8}"
          f"{'t перебор, мкс':>16}{'t мемо, мкс':>13}{'t таблица, мкс':>16}")
    for r in res["edit_naive"]:
        print(f"  {r['L']:>3}{r['distance']:>4}{r['calls_naive']:>19}{r['calls_memo']:>16}{r['cells']:>8}"
              f"{r['t_naive'] * 1e6:>16.1f}{r['t_memo'] * 1e6:>13.1f}{r['t_table'] * 1e6:>16.1f}")

    print("\nТаблица 5. Рост времени ДП с длиной строк L (мс)")
    print(f"  {'L':>4}{'клеток':>9}{'таблица':>10}{'две строки':>12}")
    for r in res["edit_scaling"]:
        print(f"  {r['L']:>4}{r['cells']:>9}{r['t_table'] * 1e3:>10.2f}{r['t_two_rows'] * 1e3:>12.2f}")

    print("\nТаблица 6 (Г6). Мемоизация: собственный декоратор / lru_cache / явный словарь")
    print("  Задача А (время в мкс, пустой кэш)")
    print(f"  {'n':>5}{'own':>10}{'lru':>10}{'словарь':>10}{'обращений own':>15}{'вычислений':>12}")
    for r in res["cache"]["stairs"]:
        print(f"  {r['n']:>5}{r['t_own'] * 1e6:>10.2f}{r['t_lru'] * 1e6:>10.2f}{r['t_dict'] * 1e6:>10.2f}"
              f"{r['calls_own']:>15}{r['misses_own']:>12}")
    print("  Задача Б (время в мс, пустой кэш)")
    print(f"  {'L':>5}{'own':>10}{'lru':>10}{'словарь':>10}{'таблица':>10}{'обращений':>11}{'вычислений':>12}")
    for r in res["cache"]["edit"]:
        print(f"  {r['L']:>5}{r['t_own'] * 1e3:>10.2f}{r['t_lru'] * 1e3:>10.2f}{r['t_dict'] * 1e3:>10.2f}"
              f"{r['t_table'] * 1e3:>10.2f}{r['calls_own']:>11}{r['misses_own']:>12}")


def main():
    print("Лабораторная работа № 7, вариант 13")
    print(system_info())
    print(f"Задача А: лестница с шагами 1, 2, 3, n = {STAIRS_N[0]}...{STAIRS_N[-1]}; "
          f"задача Б: редакционное расстояние, строки длиной {STRING_LEN}")
    self_test()
    demo_small()

    res = {}
    print("\nЗадача А ...", flush=True)
    res["stairs"] = experiment_stairs()
    res["fit"] = fit_base(res["stairs"])
    print("  поиск предела наивной рекурсии:", flush=True)
    res["limit"] = experiment_limit(res["stairs"][-1]["n"], res["stairs"][-1]["t_naive"])
    print("Задача Б ...", flush=True)
    res["edit_main"] = experiment_edit_main()
    res["edit_naive"] = experiment_edit_naive()
    res["edit_scaling"] = experiment_edit_scaling()
    print("Г6: сравнение мемоизаций ...", flush=True)
    res["cache"] = experiment_cache()
    print_report(res)
    make_plots(res)
    print("\nГрафики сохранены: lr7_plot_stairs.png, lr7_plot_edit.png, lr7_plot_cache.png")
    if "--json" in sys.argv:                        # служебный вывод для отчёта
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
