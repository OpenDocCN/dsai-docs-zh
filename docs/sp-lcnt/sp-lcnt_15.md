# 3.1 Python 中的统计学

# 3.1 Python 中的统计学

In [1]:

```py
%matplotlib inline
import numpy as np 
```

> **作者** : Gaël Varoquaux

**必要条件**

*   标准 Python 科学计算环境 (numpy, scipy, matplotlib)
*   [Pandas](http://pandas.pydata.org/)
*   [Statsmodels](http://statsmodels.sourceforge.net/)
*   [Seaborn](http://stanford.edu/~mwaskom/software/seaborn/)

要安装 Python 及这些依赖，推荐下载[Anaconda Python](http://continuum.io/downloads) 或 [Enthought Canopy](https://store.enthought.com/), 如果你使用 Ubuntu 或其他 linux 更应该使用包管理器。

**也可以看一下: Python 中的贝叶斯统计**

本章并不会涉及贝叶斯统计工具。适用于贝叶斯模型的是[PyMC](http://pymc-devs.github.io/pymc), 在 Python 中实现了概率编程语言。

**为什么统计学要用 Python?**

R 是一门专注于统计学的语言。Python 是带有统计学模块的通用编程语言。R 比 Python 有更多的统计分析功能，以及专用的语法。但是，当面对构建复杂的分析管道，混合统计学以及例如图像分析、文本挖掘或者物理实验控制，Python 的富有就是物价的优势。

**内容**

*   数据表征和交互
    *   数据作为表格
    *   panda data-frame
*   假设检验: 对比两个组
    *   Student’s t-test: 最简单的统计检验
    *   配对实验: 对同一个体的重复测量
*   线性模型、多因素和方差分析
    *   用“公式” 来在 Python 中指定统计模型
    *   多元回归: 包含多元素
    *   事后假设检验: 方差分析 (ANOVA)
*   更多的可视化: 用 seaborn 来进行统计学探索
    *   配对图: 散点矩阵
    *   lmplot: 绘制一个单变量回归
*   交互作用检验

**免责声明: 性别问题**

本教程中的一些实例选自性别问题。其原因是在这种问题上这种控制的声明实际上影响了很多人。

## 3.1.1 数据表征和交互

### 3.1.1.1 数据作为表格

统计分析中我们关注的设定是通过一组不同的属性或特征来描述多个观察或样本。然后这个数据可以被视为 2D 表格，或矩阵，列是数据的不同属性，行是观察。例如包含在[examples/brain_size.csv](http://www.scipy-lectures.org/_downloads/brain_size.csv)的数据:

`"";"Gender";"FSIQ";"VIQ";"PIQ";"Weight";"Height";"MRI_Count" "1";"Female";133;132;124;"118";"64.5";816932 "2";"Male";140;150;124;".";"72.5";1001121 "3";"Male";139;123;150;"143";"73.3";1038437 "4";"Male";133;129;128;"172";"68.8";965353 "5";"Female";137;132;134;"147";"65.0";951545`

### 3.1.1.2 panda data-frame

我们将会在来自[pandas](http://pandas.pydata.org/)模块的[pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame)中存储和操作这个数据。它是电子表格程序在 Python 中的一个等价物。它与 2D `numpy`数据的区别在于列带有名字，可以在列中存储混合的数据类型，并且有精妙的选择和透视表机制。

#### 3.1.1.2.1 创建 dataframes: 读取数据文件或转化数组

**从 CSV 文件读取**: 使用上面的 CSV 文件，给出了大脑大小重量和 IQ (Willerman et al. 1991) 的观察值 , 数据混合了数量值和类型值:

In [3]:

```py
import pandas
data = pandas.read_csv('examples/brain_size.csv', sep=';', na_values=".")
data 
```

Out[3]:

|  | Unnamed: 0 | Gender | FSIQ | VIQ | PIQ | Weight | Height | MRI_Count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | Female | 133 | 132 | 124 | 118 | 64.5 | 816932 |
| 1 | 2 | Male | 140 | 150 | 124 | NaN | 72.5 | 1001121 |
| 2 | 3 | Male | 139 | 123 | 150 | 143 | 73.3 | 1038437 |
| 3 | 4 | Male | 133 | 129 | 128 | 172 | 68.8 | 965353 |
| 4 | 5 | Female | 137 | 132 | 134 | 147 | 65.0 | 951545 |
| 5 | 6 | Female | 99 | 90 | 110 | 146 | 69.0 | 928799 |
| 6 | 7 | Female | 138 | 136 | 131 | 138 | 64.5 | 991305 |
| 7 | 8 | Female | 92 | 90 | 98 | 175 | 66.0 | 854258 |
| 8 | 9 | Male | 89 | 93 | 84 | 134 | 66.3 | 904858 |
| 9 | 10 | Male | 133 | 114 | 147 | 172 | 68.8 | 955466 |
| 10 | 11 | Female | 132 | 129 | 124 | 118 | 64.5 | 833868 |
| 11 | 12 | Male | 141 | 150 | 128 | 151 | 70.0 | 1079549 |
| 12 | 13 | Male | 135 | 129 | 124 | 155 | 69.0 | 924059 |
| 13 | 14 | Female | 140 | 120 | 147 | 155 | 70.5 | 856472 |
| 14 | 15 | Female | 96 | 100 | 90 | 146 | 66.0 | 878897 |
| 15 | 16 | Female | 83 | 71 | 96 | 135 | 68.0 | 865363 |
| 16 | 17 | Female | 132 | 132 | 120 | 127 | 68.5 | 852244 |
| 17 | 18 | Male | 100 | 96 | 102 | 178 | 73.5 | 945088 |
| 18 | 19 | Female | 101 | 112 | 84 | 136 | 66.3 | 808020 |
| 19 | 20 | Male | 80 | 77 | 86 | 180 | 70.0 | 889083 |
| 20 | 21 | Male | 83 | 83 | 86 | NaN | NaN | 892420 |
| 21 | 22 | Male | 97 | 107 | 84 | 186 | 76.5 | 905940 |
| 22 | 23 | Female | 135 | 129 | 134 | 122 | 62.0 | 790619 |
| 23 | 24 | Male | 139 | 145 | 128 | 132 | 68.0 | 955003 |
| 24 | 25 | Female | 91 | 86 | 102 | 114 | 63.0 | 831772 |
| 25 | 26 | Male | 141 | 145 | 131 | 171 | 72.0 | 935494 |
| 26 | 27 | Female | 85 | 90 | 84 | 140 | 68.0 | 798612 |
| 27 | 28 | Male | 103 | 96 | 110 | 187 | 77.0 | 1062462 |
| 28 | 29 | Female | 77 | 83 | 72 | 106 | 63.0 | 793549 |
| 29 | 30 | Female | 130 | 126 | 124 | 159 | 66.5 | 866662 |
| 30 | 31 | Female | 133 | 126 | 132 | 127 | 62.5 | 857782 |
| 31 | 32 | Male | 144 | 145 | 137 | 191 | 67.0 | 949589 |
| 32 | 33 | Male | 103 | 96 | 110 | 192 | 75.5 | 997925 |
| 33 | 34 | Male | 90 | 96 | 86 | 181 | 69.0 | 879987 |
| 34 | 35 | Female | 83 | 90 | 81 | 143 | 66.5 | 834344 |
| 35 | 36 | Female | 133 | 129 | 128 | 153 | 66.5 | 948066 |
| 36 | 37 | Male | 140 | 150 | 124 | 144 | 70.5 | 949395 |
| 37 | 38 | Female | 88 | 86 | 94 | 139 | 64.5 | 893983 |
| 38 | 39 | Male | 81 | 90 | 74 | 148 | 74.0 | 930016 |
| 39 | 40 | Male | 89 | 91 | 89 | 179 | 75.5 | 935863 |

> **分割符** 它是 CSV 文件，但是分割符是”;”
> 
> **缺失值** CSV 中的第二个个体的 weight 是缺失的。如果我们没有指定缺失值 (NA = not available) 标记符, 我们将无法进行统计分析。

**从数组中创建**: [pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame) 也可以视为 1D 序列, 例如数组或列表的字典，如果我们有 3 个`numpy`数组:

In [4]:

```py
import numpy as np
t = np.linspace(-6, 6, 20)
sin_t = np.sin(t)
cos_t = np.cos(t) 
```

我们可以将他们暴露为[pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame):

In [5]:

```py
pandas.DataFrame({'t': t, 'sin': sin_t, 'cos': cos_t}) 
```

Out[5]:

|  | cos | sin | t |
| --- | --- | --- | --- |
| 0 | 0.960170 | 0.279415 | -6.000000 |
| 1 | 0.609977 | 0.792419 | -5.368421 |
| 2 | 0.024451 | 0.999701 | -4.736842 |
| 3 | -0.570509 | 0.821291 | -4.105263 |
| 4 | -0.945363 | 0.326021 | -3.473684 |
| 5 | -0.955488 | -0.295030 | -2.842105 |
| 6 | -0.596979 | -0.802257 | -2.210526 |
| 7 | -0.008151 | -0.999967 | -1.578947 |
| 8 | 0.583822 | -0.811882 | -0.947368 |
| 9 | 0.950551 | -0.310567 | -0.315789 |
| 10 | 0.950551 | 0.310567 | 0.315789 |
| 11 | 0.583822 | 0.811882 | 0.947368 |
| 12 | -0.008151 | 0.999967 | 1.578947 |
| 13 | -0.596979 | 0.802257 | 2.210526 |
| 14 | -0.955488 | 0.295030 | 2.842105 |
| 15 | -0.945363 | -0.326021 | 3.473684 |
| 16 | -0.570509 | -0.821291 | 4.105263 |
| 17 | 0.024451 | -0.999701 | 4.736842 |
| 18 | 0.609977 | -0.792419 | 5.368421 |
| 19 | 0.960170 | -0.279415 | 6.000000 |

**其他输入**: [pandas](http://pandas.pydata.org/) 可以从 SQL、excel 文件或者其他格式输入数。见[pandas 文档](http://pandas.pydata.org/)。

#### 3.1.1.2.2 操作数据

`data`是[pandas.DataFrame](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html#pandas.DataFrame), 与 R 的 dataframe 类似:

In [6]:

```py
data.shape    # 40 行 8 列 
```

Out[6]:

```py
(40, 8) 
```

In [7]:

```py
data.columns  # 有列 
```

Out[7]:

```py
Index([u'Unnamed: 0', u'Gender', u'FSIQ', u'VIQ', u'PIQ', u'Weight', u'Height',
       u'MRI_Count'],
      dtype='object') 
```

In [8]:

```py
print(data['Gender'])  # 列可以用名字访问 
```

```py
0     Female
1       Male
2       Male
3       Male
4     Female
5     Female
6     Female
7     Female
8       Male
9       Male
10    Female
11      Male
12      Male
13    Female
14    Female
15    Female
16    Female
17      Male
18    Female
19      Male
20      Male
21      Male
22    Female
23      Male
24    Female
25      Male
26    Female
27      Male
28    Female
29    Female
30    Female
31      Male
32      Male
33      Male
34    Female
35    Female
36      Male
37    Female
38      Male
39      Male
Name: Gender, dtype: object 
```

In [9]:

```py
# 简单选择器
data[data['Gender'] == 'Female']['VIQ'].mean() 
```

Out[9]:

```py
109.45 
```

> **注意**: 对于一个大 dataframe 的快速预览，用它的`describe`方法: [pandas.DataFrame.describe()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.describe.html#pandas.DataFrame.describe)。

**groupby**: 根据类别变量的值拆分 dataframe:

In [10]:

```py
groupby_gender = data.groupby('Gender')
for gender, value in groupby_gender['VIQ']:
    print((gender, value.mean())) 
```

```py
('Female', 109.45)
('Male', 115.25) 
```

**groupby_gender**是一个强力的对象，暴露了结果 dataframes 组的许多操作:

In [11]:

```py
groupby_gender.mean() 
```

Out[11]:

|  | Unnamed: 0 | FSIQ | VIQ | PIQ | Weight | Height | MRI_Count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gender |
| Female | 19.65 | 111.9 | 109.45 | 110.45 | 137.200000 | 65.765000 | 862654.6 |
| Male | 21.35 | 115.0 | 115.25 | 111.60 | 166.444444 | 71.431579 | 954855.4 |

在`groupby_gender`上使用 tab-完成来查找更多。其他的常见分组函数是 median, count (对于检查不同子集的缺失值数量很有用) 或 sum。Groupby 评估是懒惰模式，因为在应用聚合函数之前不会进行什么工作。

> **练习**
> 
> *   完整人口 VIO 的平均值是多少?
> *   这项研究中包含了多少男性 / 女性?
> *   **提示** 使用‘tab 完成’来寻找可以调用的方法, 替换在上面例子中的‘mean’。
> *   对于男性和女性来说，以 log 为单位显示的 MRI count 平均值是多少?

![](img/c29e31b2.jpg)

> **注意**: 上面的绘图中使用了`groupby_gender.boxplot` (见[这个例子](http://www.scipy-lectures.org/packages/statistics/auto_examples/plot_pandas.html#example-plot-pandas-py))。

#### 3.1.1.2.3 绘制数据

Pandas 提供一些绘图工具 (`pandas.tools.plotting`, 后面使用的是 matplotlib) 来显示在 dataframes 数据的统计值:

**散点图矩阵**:

In [15]:

```py
from pandas.tools import plotting
plotting.scatter_matrix(data[['Weight', 'Height', 'MRI_Count']]) 
```

Out[15]:

```py
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x105c34810>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10a0ade10>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10a2d80d0>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x10a33b210>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10a3be450>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10a40d9d0>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x10a49dc10>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10a51f850>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10a5902d0>]], dtype=object) 
```

```py
/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
  if self._edgecolors == str('face'): 
```

![](img/s36.jpg)

In [16]:

```py
plotting.scatter_matrix(data[['PIQ', 'VIQ', 'FSIQ']]) 
```

Out[16]:

```py
array([[<matplotlib.axes._subplots.AxesSubplot object at 0x10a918b50>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10aa38710>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10ab29910>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x10ab8e790>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10ae207d0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10abbd090>],
       [<matplotlib.axes._subplots.AxesSubplot object at 0x10af140d0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10af89cd0>,
        <matplotlib.axes._subplots.AxesSubplot object at 0x10affa410>]], dtype=object) 
```

![](img/s11.jpg)

> **两个总体**
> 
> IQ 指标是双峰的, 似乎有两个子总体。
> 
> **练习**
> 
> 只绘制男性的散点图矩阵，然后是只有女性的。你是否认为 2 个子总体与性别相关?

## 3.1.2 假设检验: 比较两个组

对于简单的[统计检验](https://en.wikipedia.org/wiki/Statistical_hypothesis_testing)，我们将可以使用[scipy](http://docs.scipy.org/doc/)的子摸块[scipy.stats](http://docs.scipy.org/doc/scipy/reference/stats.html#module-scipy.stats):

In [17]:

```py
from scipy import stats 
```

> **也看一下**: Scipy 是一个很大的库。关于整个库的快速预览，可以看一下[scipy](http://nbviewer.ipython.org/github/cloga/scipy-lecture-notes_cn/blob/master/1.5.%20Scipy%EF%BC%9A%E9%AB%98%E7%BA%A7%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97.ipynb) 章节。

### 3.1.2.1 Student’s t 检验: 最简单的统计检验

#### 3.1.2.1.1 单样本 t-检验: 检验总体平均数的值

[scipy.stats.ttest_1samp()](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html#scipy.stats.ttest_1samp)检验数据总体的平均数是否可能等于给定值 (严格来说是否观察值来自于给定总体平均数的正态分布)。它返回一个[T 统计值](https://en.wikipedia.org/wiki/Student%27s_t-test)以及[p-值](https://en.wikipedia.org/wiki/P-value) (见函数的帮助):

In [18]:

```py
stats.ttest_1samp(data['VIQ'], 0) 
```

Out[18]:

```py
(30.088099970849328, 1.3289196468728067e-28) 
```

根据$10^-28$的 p-值，我们可以声称 IQ(VIQ 的测量值)总体平均数不是 0。

![](img/d8bc3e4a.jpg)

#### 3.1.2.1.2 双样本 t-检验: 检验不同总体的差异

我们已经看到男性和女性总体 VIQ 平均数是不同的。要检验这个差异是否是显著的，我们可以用[scipy.stats.ttest_ind()](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html#scipy.stats.ttest_ind)进行双样本检验:

In [19]:

```py
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq) 
```

Out[19]:

```py
(-0.77261617232750113, 0.44452876778583217) 
```

### 3.1.2.2 配对实验: 同一个体的重复测量

PIQ、VIQ 和 FSIQ 给出了 IQ 的 3 种测量值。让我检验一下 FISQ 和 PIQ 是否有显著差异。我们可以使用双样本检验:

In [20]:

```py
stats.ttest_ind(data['FSIQ'], data['PIQ']) 
```

Out[20]:

```py
(0.46563759638096403, 0.64277250094148408) 
```

使用这种方法的问题是忘记了两个观察之间有联系: FSIQ 和 PIQ 是在相同的个体上进行的测量。因此被试之间的差异是混淆的，并且可以使用"配对实验"或"[重复测量实验](https://en.wikipedia.org/wiki/Repeated_measures_design)"来消除。

In [21]:

```py
stats.ttest_rel(data['FSIQ'], data['PIQ']) 
```

Out[21]:

```py
(1.7842019405859857, 0.082172638183642358) 
```

![](img/3a6c61e4.jpg)

这等价于单样本的差异检验:

In [22]:

```py
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0) 
```

Out[22]:

```py
(1.7842019405859857, 0.082172638183642358) 
```

![](img/7dcc1b34.jpg)

T-tests 假定高斯误差。我们可以使用[威尔科克森符号秩检验](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test), 放松了这个假设:

In [23]:

```py
stats.wilcoxon(data['FSIQ'], data['PIQ']) 
```

Out[23]:

```py
(274.5, 0.10659492713506856) 
```

> **注意:** 非配对实验对应的非参数检验是[曼惠特尼 U 检验](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U), [scipy.stats.mannwhitneyu()](http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html#scipy.stats.mannwhitneyu)。
> 
> **练习**
> 
> *   检验男性和女性重量的差异。
> *   使用非参数检验来检验男性和女性 VIQ 的差异。
> 
> **结论**: 我们发现数据并不支持男性和女性 VIQ 不同的假设。

## 3.1.3 线性模型、多因素和因素分析

### 3.1.3.1 用“公式” 来在 Python 中指定统计模型

#### 3.1.3.1.1 简单线性回归

给定两组观察值，x 和 y, 我们想要检验假设 y 是 x 的线性函数，换句话说:

$y = x * coef + intercept + e$

其中$e$是观察噪音。我们将使用[statmodels module](http://statsmodels.sourceforge.net/):

*   拟合一个线性模型。我们将使用简单的策略，[普通最小二乘](https://en.wikipedia.org/wiki/Ordinary_least_squares) (OLS)。
*   检验系数是否是非 0。

![](img/97ace6a6.jpg)

首先，我们生成模型的虚拟数据:

In [9]:

```py
import numpy as np
x = np.linspace(-5, 5, 20)
np.random.seed(1)
# normal distributed noise
y = -5 + 3*x + 4 * np.random.normal(size=x.shape)
# Create a data frame containing all the relevant variables
data = pandas.DataFrame({'x': x, 'y': y}) 
```

> **Python 中的统计公式**
> 
> [见 statsmodels 文档](http://statsmodels.sourceforge.net/stable/example_formulas.html)

然后我们指定一个 OLS 模型并且拟合它:

In [10]:

```py
from statsmodels.formula.api import ols
model = ols("y ~ x", data).fit() 
```

我们可以检查 fit 产生的各种统计量:

In [26]:

```py
print(model.summary()) 
```

```py
 OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.804
Model:                            OLS   Adj. R-squared:                  0.794
Method:                 Least Squares   F-statistic:                     74.03
Date:                Wed, 18 Nov 2015   Prob (F-statistic):           8.56e-08
Time:                        17:10:03   Log-Likelihood:                -57.988
No. Observations:                  20   AIC:                             120.0
Df Residuals:                      18   BIC:                             122.0
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
Intercept     -5.5335      1.036     -5.342      0.000        -7.710    -3.357
x              2.9369      0.341      8.604      0.000         2.220     3.654
==============================================================================
Omnibus:                        0.100   Durbin-Watson:                   2.956
Prob(Omnibus):                  0.951   Jarque-Bera (JB):                0.322
Skew:                          -0.058   Prob(JB):                        0.851
Kurtosis:                       2.390   Cond. No.                         3.03
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified. 
```

> **术语**:
> 
> Statsmodels 使用统计术语: statsmodel 的 y 变量被称为‘endogenous’而 x 变量被称为 exogenous。更详细的讨论见[这里](http://statsmodels.sourceforge.net/devel/endog_exog.html)。
> 
> 为了简化，y (endogenous) 是你要尝试预测的值，而 x (exogenous) 代表用来进行这个预测的特征。
> 
> **练习**
> 
> 从以上模型中取回估计参数。**提示**: 使用 tab-完成来找到相关的属性。

#### 3.1.3.1.2 类别变量: 比较组或多个类别

让我们回到大脑大小的数据:

In [27]:

```py
data = pandas.read_csv('examples/brain_size.csv', sep=';', na_values=".") 
```

我们可以写一个比较，用线性模型比较男女 IQ:

In [28]:

```py
model = ols("VIQ ~ Gender + 1", data).fit()
print(model.summary()) 
```

```py
 OLS Regression Results                            
==============================================================================
Dep. Variable:                    VIQ   R-squared:                       0.015
Model:                            OLS   Adj. R-squared:                 -0.010
Method:                 Least Squares   F-statistic:                    0.5969
Date:                Wed, 18 Nov 2015   Prob (F-statistic):              0.445
Time:                        17:34:10   Log-Likelihood:                -182.42
No. Observations:                  40   AIC:                             368.8
Df Residuals:                      38   BIC:                             372.2
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==================================================================================
                     coef    std err          t      P>|t|      [95.0% Conf. Int.]
----------------------------------------------------------------------------------
Intercept        109.4500      5.308     20.619      0.000        98.704   120.196
Gender[T.Male]     5.8000      7.507      0.773      0.445        -9.397    20.997
==============================================================================
Omnibus:                       26.188   Durbin-Watson:                   1.709
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                3.703
Skew:                           0.010   Prob(JB):                        0.157
Kurtosis:                       1.510   Cond. No.                         2.62
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified. 
```

**特定模型的提示**

**强制类别**: ‘Gender’ 被自动识别为类别变量，因此，它的每一个不同值都被处理为不同的实体。 使用:

In [29]:

```py
model = ols('VIQ ~ C(Gender)', data).fit() 
```

可以将一个整数列强制作为类别处理。

**截距**: 我们可以在公式中用-1 删除截距，或者用+1 强制使用截距。

默认，statsmodel 将带有 K 和可能值的类别变量处理为 K-1'虚拟变量' (最后一个水平被吸收到截距项中)。在绝大多数情况下，这都是很好的默认选择 - 但是，为类别变量指定不同的编码方式也是可以的 ([`statsmodels.sourceforge.net/devel/contrasts.html)。`](http://statsmodels.sourceforge.net/devel/contrasts.html)。)

**FSIQ 和 PIQ 差异的 t-检验**

要比较不同类型的 IQ，我们需要创建一个"长形式"的表格，用一个类别变量来标识 IQ 类型:

In [30]:

```py
data_fisq = pandas.DataFrame({'iq': data['FSIQ'], 'type': 'fsiq'})
data_piq = pandas.DataFrame({'iq': data['PIQ'], 'type': 'piq'})
data_long = pandas.concat((data_fisq, data_piq))
print(data_long) 
```

```py
 iq  type
0   133  fsiq
1   140  fsiq
2   139  fsiq
3   133  fsiq
4   137  fsiq
5    99  fsiq
6   138  fsiq
7    92  fsiq
8    89  fsiq
9   133  fsiq
10  132  fsiq
11  141  fsiq
12  135  fsiq
13  140  fsiq
14   96  fsiq
15   83  fsiq
16  132  fsiq
17  100  fsiq
18  101  fsiq
19   80  fsiq
20   83  fsiq
21   97  fsiq
22  135  fsiq
23  139  fsiq
24   91  fsiq
25  141  fsiq
26   85  fsiq
27  103  fsiq
28   77  fsiq
29  130  fsiq
..  ...   ...
10  124   piq
11  128   piq
12  124   piq
13  147   piq
14   90   piq
15   96   piq
16  120   piq
17  102   piq
18   84   piq
19   86   piq
20   86   piq
21   84   piq
22  134   piq
23  128   piq
24  102   piq
25  131   piq
26   84   piq
27  110   piq
28   72   piq
29  124   piq
30  132   piq
31  137   piq
32  110   piq
33   86   piq
34   81   piq
35  128   piq
36  124   piq
37   94   piq
38   74   piq
39   89   piq

[80 rows x 2 columns] 
```

In [31]:

```py
model = ols("iq ~ type", data_long).fit()
print(model.summary()) 
```

```py
 OLS Regression Results                            
==============================================================================
Dep. Variable:                     iq   R-squared:                       0.003
Model:                            OLS   Adj. R-squared:                 -0.010
Method:                 Least Squares   F-statistic:                    0.2168
Date:                Wed, 18 Nov 2015   Prob (F-statistic):              0.643
Time:                        18:16:40   Log-Likelihood:                -364.35
No. Observations:                  80   AIC:                             732.7
Df Residuals:                      78   BIC:                             737.5
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
===============================================================================
                  coef    std err          t      P>|t|      [95.0% Conf. Int.]
-------------------------------------------------------------------------------
Intercept     113.4500      3.683     30.807      0.000       106.119   120.781
type[T.piq]    -2.4250      5.208     -0.466      0.643       -12.793     7.943
==============================================================================
Omnibus:                      164.598   Durbin-Watson:                   1.531
Prob(Omnibus):                  0.000   Jarque-Bera (JB):                8.062
Skew:                          -0.110   Prob(JB):                       0.0178
Kurtosis:                       1.461   Cond. No.                         2.62
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified. 
```

我们可以看到我们获得了与前面 t-检验相同的值，以及相同的对应 iq type 的 p-值:

In [32]:

```py
stats.ttest_ind(data['FSIQ'], data['PIQ']) 
```

Out[32]:

```py
(0.46563759638096403, 0.64277250094148408) 
```

### 3.1.3.2 多元回归: 包含多因素

考虑用 2 个变量 x 和 y 来解释变量 z 的线性模型:

$z = x \, c_1 + y \, c_2 + i + e$

这个模型可以被视为在 3D 世界中用一个平面去拟合 (x, y, z) 的点云。

![](img/81524c78.jpg)

**实例: 鸢尾花数据 (examples/iris.csv)**

萼片和花瓣的大小似乎是相关的: 越大的花越大! 但是，在不同的种之间是否有额外的系统效应?

![](img/9fbf8710.jpg)

In [33]:

```py
data = pandas.read_csv('examples/iris.csv')
model = ols('sepal_width ~ name + petal_length', data).fit()
print(model.summary()) 
```

```py
 OLS Regression Results                            
==============================================================================
Dep. Variable:            sepal_width   R-squared:                       0.478
Model:                            OLS   Adj. R-squared:                  0.468
Method:                 Least Squares   F-statistic:                     44.63
Date:                Thu, 19 Nov 2015   Prob (F-statistic):           1.58e-20
Time:                        09:56:04   Log-Likelihood:                -38.185
No. Observations:                 150   AIC:                             84.37
Df Residuals:                     146   BIC:                             96.41
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
======================================================================================
                         coef    std err          t      P>|t|      [95.0% Conf. Int.]
--------------------------------------------------------------------------------------
Intercept              2.9813      0.099     29.989      0.000         2.785     3.178
name[T.versicolor]    -1.4821      0.181     -8.190      0.000        -1.840    -1.124
name[T.virginica]     -1.6635      0.256     -6.502      0.000        -2.169    -1.158
petal_length           0.2983      0.061      4.920      0.000         0.178     0.418
==============================================================================
Omnibus:                        2.868   Durbin-Watson:                   1.753
Prob(Omnibus):                  0.238   Jarque-Bera (JB):                2.885
Skew:                          -0.082   Prob(JB):                        0.236
Kurtosis:                       3.659   Cond. No.                         54.0
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified. 
```

### 3.1.3.3 事后假设检验: 方差分析 (ANOVA))

在上面的鸢尾花例子中，在排除了萼片的影响之后，我们想要检验 versicolor 和 virginica 的花瓣长度是否有差异。这可以被公式化为检验在上面的线性模型中 versicolor 和 virginica 系数的差异 (方差分析, ANOVA)。我们写了"差异"向量的参数来估计: 我们想要用[F-检验](https://en.wikipedia.org/wiki/F-test)检验 "`name[T.versicolor] - name[T.virginica]`":

In [36]:

```py
print(model.f_test([0, 1, -1, 0])) 
```

```py
<F test: F=array([[ 3.24533535]]), p=0.073690587817, df_denom=146, df_num=1> 
```

是否差异显著？

> **练习** 回到大脑大小 + IQ 数据, 排除了大脑大小、高度和重量的影响后，检验男女的 VIQ 差异。

## 3.1.4 更多可视化: 用 seaborn 来进行统计学探索

[Seaborn](http://stanford.edu/~mwaskom/software/seaborn/) 集成了简单的统计学拟合与 pandas dataframes 绘图。

让我们考虑一个 500 个个体的工资及其它个人信息的数据 ([Berndt, ER. The Practice of Econometrics. 1991\. NY: Addison-Wesley](http://lib.stat.cmu.edu/datasets/CPS_85_Wages))。

加载并绘制工资数据的完整代码可以在[对应的例子](http://www.scipy-lectures.org/packages/statistics/auto_examples/plot_wage_data.html#example-plot-wage-data-py)中找到。

In [3]:

```py
print data 
```

```py
 EDUCATION  SOUTH  SEX  EXPERIENCE  UNION   WAGE  AGE  RACE  OCCUPATION  \
0            8      0    1          21      0   5.10   35     2           6   
1            9      0    1          42      0   4.95   57     3           6   
2           12      0    0           1      0   6.67   19     3           6   
3           12      0    0           4      0   4.00   22     3           6   
4           12      0    0          17      0   7.50   35     3           6   
5           13      0    0           9      1  13.07   28     3           6   
6           10      1    0          27      0   4.45   43     3           6   
7           12      0    0           9      0  19.47   27     3           6   
8           16      0    0          11      0  13.28   33     3           6   
9           12      0    0           9      0   8.75   27     3           6   
10          12      0    0          17      1  11.35   35     3           6   
11          12      0    0          19      1  11.50   37     3           6   
12           8      1    0          27      0   6.50   41     3           6   
13           9      1    0          30      1   6.25   45     3           6   
14           9      1    0          29      0  19.98   44     3           6   
15          12      0    0          37      0   7.30   55     3           6   
16           7      1    0          44      0   8.00   57     3           6   
17          12      0    0          26      1  22.20   44     3           6   
18          11      0    0          16      0   3.65   33     3           6   
19          12      0    0          33      0  20.55   51     3           6   
20          12      0    1          16      1   5.71   34     3           6   
21           7      0    0          42      1   7.00   55     1           6   
22          12      0    0           9      0   3.75   27     3           6   
23          11      1    0          14      0   4.50   31     1           6   
24          12      0    0          23      0   9.56   41     3           6   
25           6      1    0          45      0   5.75   57     3           6   
26          12      0    0           8      0   9.36   26     3           6   
27          10      0    0          30      0   6.50   46     3           6   
28          12      0    1           8      0   3.35   26     3           6   
29          12      0    0           8      0   4.75   26     3           6   
..         ...    ...  ...         ...    ...    ...  ...   ...         ...   
504         17      0    1          10      0  11.25   33     3           5   
505         16      0    1          10      1   6.67   32     3           5   
506         16      0    1          17      0   8.00   39     2           5   
507         18      0    0           7      0  18.16   31     3           5   
508         16      0    1          14      0  12.00   36     3           5   
509         16      0    1          22      1   8.89   44     3           5   
510         17      0    1          14      0   9.50   37     3           5   
511         16      0    0          11      0  13.65   33     3           5   
512         18      0    0          23      1  12.00   47     3           5   
513         12      0    0          39      1  15.00   57     3           5   
514         16      0    0          15      0  12.67   37     3           5   
515         14      0    1          15      0   7.38   35     2           5   
516         16      0    0          10      0  15.56   32     3           5   
517         12      1    1          25      0   7.45   43     3           5   
518         14      0    1          12      0   6.25   32     3           5   
519         16      1    1           7      0   6.25   29     2           5   
520         17      0    0           7      1   9.37   30     3           5   
521         16      0    0          17      0  22.50   39     3           5   
522         16      0    0          10      1   7.50   32     3           5   
523         17      1    0           2      0   7.00   25     3           5   
524          9      1    1          34      1   5.75   49     1           5   
525         15      0    1          11      0   7.67   32     3           5   
526         15      0    0          10      0  12.50   31     3           5   
527         12      1    0          12      0  16.00   30     3           5   
528         16      0    1           6      1  11.79   28     3           5   
529         18      0    0           5      0  11.36   29     3           5   
530         12      0    1          33      0   6.10   51     1           5   
531         17      0    1          25      1  23.25   48     1           5   
532         12      1    0          13      1  19.88   31     3           5   
533         16      0    0          33      0  15.38   55     3           5   

     SECTOR  MARR  
0         1     1  
1         1     1  
2         1     0  
3         0     0  
4         0     1  
5         0     0  
6         0     0  
7         0     0  
8         1     1  
9         0     0  
10        0     1  
11        1     0  
12        0     1  
13        0     0  
14        0     1  
15        2     1  
16        0     1  
17        1     1  
18        0     0  
19        0     1  
20        1     1  
21        1     1  
22        0     0  
23        0     1  
24        0     1  
25        1     1  
26        1     1  
27        0     1  
28        1     1  
29        0     1  
..      ...   ...  
504       0     0  
505       0     0  
506       0     1  
507       0     1  
508       0     1  
509       0     1  
510       0     1  
511       0     1  
512       0     1  
513       0     1  
514       0     1  
515       0     0  
516       0     0  
517       0     0  
518       0     1  
519       0     1  
520       0     1  
521       1     1  
522       0     1  
523       0     1  
524       0     1  
525       0     1  
526       0     0  
527       0     1  
528       0     0  
529       0     0  
530       0     1  
531       0     1  
532       0     1  
533       1     1  

[534 rows x 11 columns] 
```

### 3.1.4.1 配对图: 散点矩阵

使用[seaborn.pairplot()](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.pairplot.html#seaborn.pairplot)来显示散点矩阵我们可以很轻松的对连续变量之间的交互有一个直觉:

In [4]:

```py
import seaborn
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg') 
```

Out[4]:

```py
<seaborn.axisgrid.PairGrid at 0x107feb850> 
```

```py
/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/matplotlib/collections.py:590: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
  if self._edgecolors == str('face'): 
```

![](img/s39.jpg)

可以用颜色来绘制类别变量:

In [5]:

```py
seaborn.pairplot(data, vars=['WAGE', 'AGE', 'EDUCATION'], kind='reg', hue='SEX') 
```

Out[5]:

```py
<seaborn.axisgrid.PairGrid at 0x107feb650> 
```

![](img/s44.jpg)

**看一下并感受一些 matplotlib 设置**

Seaborn 改变了 matplotlib 的默认图案以便获得更"现代"、更"类似 Excel"的外观。它是通过 import 来实现的。重置默认设置可以使用:

In [8]:

```py
from matplotlib import pyplot as plt
plt.rcdefaults() 
```

要切换回 seaborn 设置, 或者更好理解 seaborn 中的样式, 见[seaborn 文档中的相关部分](http://stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html)。

### 3.1.4.2\. lmplot: 绘制一个单变量回归

回归捕捉了一个变量与另一个变量的关系，例如薪水和教育，可以用[seaborn.lmplot()](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.lmplot.html#seaborn.lmplot)来绘制:

In [6]:

```py
seaborn.lmplot(y='WAGE', x='EDUCATION', data=data) 
```

Out[6]:

```py
<seaborn.axisgrid.FacetGrid at 0x108db6050> 
```

![](img/s10.jpg)

**稳健回归**

在上图中，有一些数据点偏离了右侧的主要云，他们可能是异常值，对总体没有代表性，但是，推动了回归。

要计算对异常值不敏感的回归，必须使用[稳健模型](https://en.wikipedia.org/wiki/Robust_statistics)。在 seaborn 的绘图函数中可以使用`robust=True`，或者在 statsmodels 用"稳健线性回归"`statsmodels.formula.api.rlm()`来替换 OLS。

## 3.1.5 交互作用检验

![](img/43e176b1.jpg)

是否教育对工资的提升在男性中比女性中更多?

上图来自两个不同的拟合。我们需要公式化一个简单的模型来检验总体倾斜的差异。这通过"交互作用"来完成。

In [22]:

```py
result = ols(formula='WAGE ~ EDUCATION + C(SEX) + EDUCATION * C(SEX)', data=data).fit()    
print(result.summary()) 
```

```py
 OLS Regression Results                            
==============================================================================
Dep. Variable:                   WAGE   R-squared:                       0.190
Model:                            OLS   Adj. R-squared:                  0.186
Method:                 Least Squares   F-statistic:                     41.50
Date:                Thu, 19 Nov 2015   Prob (F-statistic):           4.24e-24
Time:                        12:06:38   Log-Likelihood:                -1575.0
No. Observations:                 534   AIC:                             3158.
Df Residuals:                     530   BIC:                             3175.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=========================================================================================
                            coef    std err          t      P>|t|      [95.0% Conf. Int.]
-----------------------------------------------------------------------------------------
Intercept                 1.1046      1.314      0.841      0.401        -1.476     3.685
C(SEX)[T.1]              -4.3704      2.085     -2.096      0.037        -8.466    -0.274
EDUCATION                 0.6831      0.099      6.918      0.000         0.489     0.877
EDUCATION:C(SEX)[T.1]     0.1725      0.157      1.098      0.273        -0.136     0.481
==============================================================================
Omnibus:                      208.151   Durbin-Watson:                   1.863
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1278.081
Skew:                           1.587   Prob(JB):                    2.94e-278
Kurtosis:                       9.883   Cond. No.                         170.
==============================================================================

Warnings:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified. 
```

我们可以得出结论教育对男性的益处大于女性吗？

**带回家的信息**

*   假设检验和 p-值告诉你影响 / 差异的**显著性**
*   **公式** (带有类别变量) 让你可以表达你数据中的丰富联系
*   **可视化**数据和简单模型拟合很重要!
*   **条件化** (添加可以解释所有或部分方差的因素) 在改变交互作用建模方面非常重要。

## 3.1.6 完整例子

### [3.1.6.1 例子](http://www.scipy-lectures.org/packages/statistics/auto_examples/index.html)

#### [3.1.6.1.1 代码例子](http://www.scipy-lectures.org/packages/statistics/auto_examples/index.html#code-examples)

#### [3.1.6.1.2 课程练习的答案](http://www.scipy-lectures.org/packages/statistics/auto_examples/index.html#solutions-to-the-exercises-of-the-course)