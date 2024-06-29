
# 3.2 Sympy：Python 中的符号数学

> **作者** : Fabian Pedregosa

**目的**

*   从任意的精度评估表达式。
*   在符号表达式上进行代数运算。
*   用符号表达式进行基本的微积分任务 (极限、微分法和积分法)。
*   求解多项式和超越方程。
*   求解一些微分方程。

为什么是 SymPy? SymPy 是符号数学的 Python 库。它的目的是成为 Mathematica 或 Maple 等系统的替代品，同时让代码尽可能简单并且可扩展。SymPy 完全是用 Python 写的，并不需要外部的库。

Sympy 文档及库安装见[`www.sympy.org/`](http://www.sympy.org/)

**章节内容**

*   SymPy 第一步
    *   使用 SymPy 作为计算器
    *   练习
    *   符号
*   代数运算
    *   展开
    *   化简
*   微积分
    *   极限
    *   微分法
    *   序列扩展
    *   积分法
    *   练习
*   方程求解
    *   练习
*   线性代数
    *   矩阵
    *   微分方程

## 3.2.1 SymPy 第一步

### 3.2.1.1 使用 SymPy 作为计算器

SymPy 定义了三种数字类型：实数、有理数和整数。

有理数类将有理数表征为两个整数对: 分子和分母，因此 Rational(1,2)代表 1/2, Rational(5,2)代表 5/2 等等:

In [2]:

```py
from sympy import *
a = Rational(1,2) 
```

In [2]:

```py
a 
```

Out[2]:

```py
1/2 
```

In [3]:

```py
a*2 
```

Out[3]:

```py
1 
```

SymPy 在底层使用 mpmath, 这使它可以用任意精度的算术进行计算。这样，一些特殊的常数，比如 e, pi, oo (无限), 可以被作为符号处理并且可以以任意精度来评估:

In [4]:

```py
pi**2 
```

Out[4]:

```py
pi**2 
```

In [5]:

```py
pi.evalf() 
```

Out[5]:

```py
3.14159265358979 
```

In [6]:

```py
(pi + exp(1)).evalf() 
```

Out[6]:

```py
5.85987448204884 
```

如你所见，将表达式评估为浮点数。

也有一个类代表数学的无限, 称为 oo:

In [7]:

```py
oo > 99999 
```

Out[7]:

```py
True 
```

In [8]:

```py
oo + 1 
```

Out[8]:

```py
oo 
```

### 3.2.1.2 练习

*   计算 $\sqrt{2}$ 小数点后一百位。
*   用有理数算术计算 1/2 + 1/3 in rational arithmetic.

### 3.2.1.3 符号

与其他计算机代数系统不同，在 SymPy 你需要显性声明符号变量:

In [4]:

```py
from sympy import *
x = Symbol('x')
y = Symbol('y') 
```

然后你可以计算他们:

In [10]:

```py
x + y + x - y 
```

Out[10]:

```py
2*x 
```

In [11]:

```py
(x + y)**2 
```

Out[11]:

```py
(x + y)**2 
```

符号可以使用一些 Python 操作符操作: +, -, *, ** (算术), &, |, ~ , >>, << (布尔逻辑).

**打印** 这里我们使用下列设置打印

In [ ]:

```py
sympy.init_printing(use_unicode=False, wrap_line=True) 
```

## 3.2.2 代数运算

SymPy 可以进行强大的代数运算。我们将看一下最常使用的：展开和化简。

### 3.2.2.1 展开

使用这个模块展开代数表达式。它将试着密集的乘方和相乘:

In [13]:

```py
expand((x + y)**3) 
```

Out[13]:

```py
x**3 + 3*x**2*y + 3*x*y**2 + y**3 
```

In [14]:

```py
3*x*y**2 + 3*y*x**2 + x**3 + y**3 
```

Out[14]:

```py
x**3 + 3*x**2*y + 3*x*y**2 + y**3 
```

可以通过关键词的形式使用更多的选项:

In [15]:

```py
expand(x + y, complex=True) 
```

Out[15]:

```py
re(x) + re(y) + I*im(x) + I*im(y) 
```

In [16]:

```py
I*im(x) + I*im(y) + re(x) + re(y) 
```

Out[16]:

```py
re(x) + re(y) + I*im(x) + I*im(y) 
```

In [17]:

```py
expand(cos(x + y), trig=True) 
```

Out[17]:

```py
-sin(x)*sin(y) + cos(x)*cos(y) 
```

In [18]:

```py
cos(x)*cos(y) - sin(x)*sin(y) 
```

Out[18]:

```py
-sin(x)*sin(y) + cos(x)*cos(y) 
```

## 3.2.2.2 化简

如果可以将表达式转化为更简单的形式，可以使用化简:

In [19]:

```py
simplify((x + x*y) / x) 
```

Out[19]:

```py
y + 1 
```

化简是一个模糊的术语，更准确的词应该是：powsimp (指数化简)、 trigsimp (三角表达式)、logcombine、radsimp 一起。

**练习**

*   计算$(x+y)⁶$的展开。
*   化简三角表达式$ \sin(x) / \cos(x)$

## 3.2.3 微积分

### 3.2.3.1 极限

在 SymPy 中使用极限很简单，允许语法 limit(function, variable, point), 因此要计算 f(x)类似$x \rightarrow 0$, 你应该使用 limit(f, x, 0):

In [5]:

```py
limit(sin(x)/x, x, 0) 
```

Out[5]:

```py
1 
```

你也可以计算一下在无限时候的极限:

In [6]:

```py
limit(x, x, oo) 
```

Out[6]:

```py
oo 
```

In [7]:

```py
limit(1/x, x, oo) 
```

Out[7]:

```py
0 
```

In [8]:

```py
limit(x**x, x, 0) 
```

Out[8]:

```py
1 
```

### 3.2.3.2 微分法

你可以使用`diff(func, var)`微分任何 SymPy 表达式。例如:

In [9]:

```py
diff(sin(x), x) 
```

Out[9]:

```py
cos(x) 
```

In [10]:

```py
diff(sin(2*x), x) 
```

Out[10]:

```py
2*cos(2*x) 
```

In [11]:

```py
diff(tan(x), x) 
```

Out[11]:

```py
tan(x)**2 + 1 
```

你可以用下列方法检查是否正确:

In [12]:

```py
limit((tan(x+y) - tan(x))/y, y, 0) 
```

Out[12]:

```py
tan(x)**2 + 1 
```

可以用`diff(func, var, n)`方法来计算更高的导数:

In [13]:

```py
diff(sin(2*x), x, 1) 
```

Out[13]:

```py
2*cos(2*x) 
```

In [14]:

```py
diff(sin(2*x), x, 2) 
```

Out[14]:

```py
-4*sin(2*x) 
```

In [15]:

```py
diff(sin(2*x), x, 3) 
```

Out[15]:

```py
-8*cos(2*x) 
```

### 3.2.3.3 序列展开

SymPy 也知道如何计算一个表达式在一个点的 Taylor 序列。使用`series(expr, var)`:

In [16]:

```py
series(cos(x), x) 
```

Out[16]:

```py
1 - x**2/2 + x**4/24 + O(x**6) 
```

In [17]:

```py
series(1/cos(x), x) 
```

Out[17]:

```py
1 + x**2/2 + 5*x**4/24 + O(x**6) 
```

**练习**

计算$\lim_{x\rightarrow 0} \sin(x)/x$

计算`log(x)`对于 x 的导数。

### 3.2.3.4 积分法

SymPy 支持超验基础和特殊函数的无限和有限积分，通过`integrate()` 功能, 使用了强大的扩展的 Risch-Norman 算法和启发式和模式匹配。你可以积分基本函数:

In [18]:

```py
integrate(6*x**5, x) 
```

Out[18]:

```py
x**6 
```

In [19]:

```py
integrate(sin(x), x) 
```

Out[19]:

```py
-cos(x) 
```

In [20]:

```py
integrate(log(x), x) 
```

Out[20]:

```py
x*log(x) - x 
```

In [21]:

```py
integrate(2*x + sinh(x), x) 
```

Out[21]:

```py
x**2 + cosh(x) 
```

也可以很简单的处理特殊函数:

In [22]:

```py
integrate(exp(-x**2)*erf(x), x) 
```

Out[22]:

```py
sqrt(pi)*erf(x)**2/4 
```

也可以计算一下有限积分:

In [23]:

```py
integrate(x**3, (x, -1, 1)) 
```

Out[23]:

```py
0 
```

In [24]:

```py
integrate(sin(x), (x, 0, pi/2)) 
```

Out[24]:

```py
1 
```

In [25]:

```py
integrate(cos(x), (x, -pi/2, pi/2)) 
```

Out[25]:

```py
2 
```

不标准积分也支持:

In [26]:

```py
integrate(exp(-x), (x, 0, oo)) 
```

Out[26]:

```py
1 
```

In [27]:

```py
integrate(exp(-x**2), (x, -oo, oo)) 
```

Out[27]:

```py
sqrt(pi) 
```

#### 3.2.3.5 练习

### 3.2.4 方程求解

SymPy 可以求解线性代数方程，一个或多个变量:

In [28]:

```py
solve(x**4 - 1, x) 
```

Out[28]:

```py
[-1, 1, -I, I] 
```

如你所见，第一个参数是假设等于 0 的表达式。它可以解一个很大的多项式方程，也可以有能力求解多个方程，可以将各自的多个变量作为元组以第二个参数给出:

In [29]:

```py
solve([x + 5*y - 2, -3*x + 6*y - 15], [x, y]) 
```

Out[29]:

```py
{x: -3, y: 1} 
```

也直接求解超越方程（有限的）:

In [30]:

```py
solve(exp(x) + 1, x) 
```

Out[30]:

```py
[I*pi] 
```

多项式方程的另一个应用是`factor`。`factor`将多项式因式分解为可化简的项，并且可以计算不同域的因式:

In [31]:

```py
f = x**4 - 3*x**2 + 1
factor(f) 
```

Out[31]:

```py
(x**2 - x - 1)*(x**2 + x - 1) 
```

In [32]:

```py
factor(f, modulus=5) 
```

Out[32]:

```py
(x - 2)**2*(x + 2)**2 
```

SymPy 也可以解布尔方程，即，判断一个布尔表达式是否满足。对于这个情况，我们可以使用`satisfiable`函数:

In [33]:

```py
satisfiable(x & y) 
```

Out[33]:

```py
{x: True, y: True} 
```

这告诉我们`(x & y)`是真，当 x 和 y 都是 True 的时候。如果一个表达式不是 True，即它的任何参数值都无法使表达式为真，那么它将返回 False:

In [34]:

```py
satisfiable(x & ~x) 
```

Out[34]:

```py
False 
```

### 3.2.4.1 练习

*   求解系统方程$x + y = 2$, $2\cdot x + y = 0$
*   是否存在布尔值，使$(~x | y) & (~y | x)$为真?

### 3.2.5 线性代数

#### 3.2.5.1 矩阵

矩阵通过 Matrix 类的一个实例来创建:

In [35]:

```py
from sympy import Matrix
Matrix([[1,0], [0,1]]) 
```

Out[35]:

```py
Matrix([
[1, 0],
[0, 1]]) 
```

与 NumPy 数组不同，你也可以在里面放入符号:

In [36]:

```py
x = Symbol('x')
y = Symbol('y')
A = Matrix([[1,x], [y,1]])
A 
```

Out[36]:

```py
Matrix([
[1, x],
[y, 1]]) 
```

In [37]:

```py
A**2 
```

Out[37]:

```py
Matrix([
[x*y + 1,     2*x],
[    2*y, x*y + 1]]) 
```

### 3.2.5.2 微分方程

SymPy 可以解 (一些) 常规微分。要求解一个微分方程，使用`dsolve`。首先，通过传递 cls=Function 来创建一个未定义的符号函数:

In [38]:

```py
f, g = symbols('f g', cls=Function) 
```

f 和 g 是未定义函数。我们可以调用 f(x), 并且它可以代表未知的函数:

In [39]:

```py
f(x) 
```

Out[39]:

```py
f(x) 
```

In [40]:

```py
f(x).diff(x, x) + f(x) 
```

Out[40]:

```py
f(x) + Derivative(f(x), x, x) 
```

In [41]:

```py
dsolve(f(x).diff(x, x) + f(x), f(x)) 
```

Out[41]:

```py
f(x) == C1*sin(x) + C2*cos(x) 
```

关键词参数可以向这个函数传递，以便帮助确认是否找到最适合的解决系统。例如，你知道它是独立的方程，你可以使用关键词 hint=’separable’来强制`dsolve`来将它作为独立方程来求解:

In [42]:

```py
dsolve(sin(x)*cos(f(x)) + cos(x)*sin(f(x))*f(x).diff(x), f(x), hint='separable') 
```

Out[42]:

```py
[f(x) == -asin(sqrt(C1/cos(x)**2 + 1)) + pi,
 f(x) == asin(sqrt(C1/cos(x)**2 + 1)) + pi,
 f(x) == -asin(sqrt(C1/cos(x)**2 + 1)),
 f(x) == asin(sqrt(C1/cos(x)**2 + 1))] 
```

**练习**

*   求解 Bernoulli 微分方程

    $x \frac{d f(x)}{x} + f(x) - f(x)²=0$

*   使用 hint=’Bernoulli’求解相同的公式。可以观察到什么?