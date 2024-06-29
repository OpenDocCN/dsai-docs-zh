# 1.1 科学计算工具及流程

# 1.1 科学计算工具及流程

> 作者 : Fernando Perez, Emmanuelle Gouillart, Gaël Varoquaux, Valentin Haenel

## 1.1.1 为什么是 Python？

### 1.1.1.1 科学家的需求

*   获得数据（模拟，实验控制）
*   操作及处理数据
*   可视化结果... 理解我们在做什么！
*   沟通结果：生成报告或出版物的图片，写报告

### 1.1.1.2 要求

*   对于经典的数学方法及基本的方法，有丰富的现成工具：我们不希望重新编写程序去画出曲线、傅立叶变换或者拟合算法。不要重复发明轮子！
*   易于学习：计算机科学不是我们的工作也不是我们的教育背景。我们想要在几分钟内画出曲线，平滑一个信号或者做傅立叶变换，
*   可以方便的与合作者、学生、客户进行交流，代码可以存在于实验室或公司里面：代码的可读性应该像书一样。因此，这种语言应该包含尽可能少的语法符号或者不必要的常规规定，使来自数学或科学领域读者愉悦的理解这些代码。
*   语言高效，执行快...但是不需要是非常快的代码，因为如果我们花费了太多的时间来写代码，非常快的代码也是无用的。
*   一个单一的语言/环境做所有事，如果可能的话，避免每个新问题都要学习新软件

### 1.1.1.3 现有的解决方案

科学家用哪种解决方案进行工作？

### 编译语言：C、C++、Fortran 等。

*   优势：

    *   非常快。极度优化的编译器。对于大量的计算来说，很难比这些语言的性能更好。
    *   一些非常优化的科学计算包。比如：BLAS（向量/矩阵操作）
*   不足：

    *   使用起来令人痛苦：开发过程中没有任何互动，强制编译步骤，啰嗦的语法（&, ::, }}, ; 等），手动内存管理（在 C 中非常棘手）。对于非计算机学家他们是**艰深的语言**。

### 脚本语言：Matlab

*   优势：

    *   对不同的领域的多种算法都有非常的类库。执行很快，因为这些类库通常使用编译语言写的。
    *   友好的开发环境：完善的、组织良好的帮助，整合的编辑器等
    *   有商业支持
*   不足：

    *   基础语言非常欠缺，会限制高级用户
    *   不是免费的

### 其他脚本语言：Scilab、Octave、Igor、R、IDL 等。

*   优势：

    *   开源、免费，或者至少比 Matlba 便宜。
    *   一些功能非常高级（R 的统计，Igor 的图形等。）
*   不足：

    *   比 Matlab 更少的可用算法，语言也并不更高级
    *   一些软件更专注于一个领域。比如，Gnuplot 或 xmgrace 画曲线。这些程序非常强大，但是他们只限定于一个单一用途，比如作图。

### 那 Python 呢？

*   优势：

    *   非常丰富的科学计算包（尽管比 Matlab 少一些）
    *   精心设计的语言，允许写出可读性非常好并且结构良好的代码：我们“按照我们所想去写代码”。
    *   对于科学计算外的其他任务也有许多类库（网站服务器管理，串口接收等等。）
    *   免费的开源软件，广泛传播，有一个充满活力的社区。
*   不足：

    *   不太友好的开发环境，比如与 Matlab 相比。（更加极客向）。
    *   并不是在其他专业软件或工具箱中可以找到算法都可以找到

## 1.1.2 Python 科学计算的构成

与 Matlba，Scilab 或者 R 不同，Python 并没有预先绑定的一组科学计算模块。下面是可以组合起来获得科学计算环境的基础的组件。

*   **Python**，通用的现代计算语言

    *   Python 语言：数据类型（字符 string，整型 int），流程控制，数据集合（列表 list，字典 dict），模式等等。
    *   标准库及模块
    *   用 Pyhon 写的大量专业模块及应用：网络协议、网站框架等...以及科学计算。
    *   开发工具（自动测试，文档生成）
*   **IPython**, 高级的**Python Shell** [`ipython.org/`](http://ipython.org/) ![ipython](img/141bef9f.jpg)

*   **Numpy** : 提供了强大数值数组对象以及程序去操作它们。[`www.numpy.org/`](http://www.numpy.org/)

*   **Scipy** : 高级的数据处理程序。优化、回归插值等[`www.scipy.org/`](http://www.scipy.org/)

*   **Matplotlib** : 2D 可视化，“出版级”的图表[`matplotlib.sourceforge.net/`](http://matplotlib.sourceforge.net/) ![Matplotlib](img/8a3a1259.jpg)

*   **Mayavi** : 3D 可视化[`code.enthought.com/projects/mayavi/`](http://code.enthought.com/projects/mayavi/) ![Mayavi](img/674b03bb.jpg)

## 1.1.3 交互工作流：IPython 和文本编辑器

**测试和理解算法的交互工作**：在这个部分我们描述一下用[IPython](http://ipython.org/)的交互工作流来方便的研究和理解算法。

Python 是一门通用语言。与其他的通用语言一样，没有一个绝对权威的工作环境，也不止一种方法使用它。尽管这对新人来说不太好找到适合自己的方式，但是，这使得 Python 被用于在网站服务器或嵌入设备中编写程序。

> **本部分的参考文档**：
> 
> **IPython 用户手册**：[`ipython.org/ipython-doc/dev/index.html`](http://ipython.org/ipython-doc/dev/index.html)

### 1.1.3.1 命令行交互

启动 ipython:

In [1]:

```py
print('Hello world') 
```

```py
Hello world 
```

在对象后使用？运算符获得帮助:

```py
In [2]: print
Type:          builtin_function_or_method
Base Class:    <type ’builtin_function_or_method’>
String Form:   <built-in function print>
Namespace:     Python builtin
Docstring:
    print(value, ..., sep=’ ’, end=’\n’, file=sys.stdout)
    Prints the values to a stream, or to sys.stdout by default.
    Optional keyword arguments:
    file: a file-like object (stream); defaults to the current sys.stdout.
    sep:  string inserted between values, default a space.
    end:  string appended after the last value, default a newline. 
```

### 1.1.3.2 在编辑器中详尽描述算法

在文本编辑器中，创建一个 my*file.py 文件。在 EPD（[Enthought Python Distribution](https://www.enthought.com/products/epd/)）中，你可以从开始按钮使用 _Scite*。在[Python(x,y)](https://code.google.com/p/pythonxy/)中, 你可以使用 Spyder。在 Ubuntu 中, 如果你还没有最喜欢的编辑器，我们建议你安装[Stani’s Python editor](http://sourceforge.net/projects/spe/)。在这个文件中，输入如下行：

```py
s = 'Hello world'
print(s) 
```

现在，你可以在 IPython 中运行它，并研究产生的变量：

In [2]:

```py
%run my_file.py 
```

```py
Hello world 
```

In [3]:

```py
s 
```

Out[3]:

```py
'Hello world' 
```

In [4]:

```py
%whos 
```

```py
Variable   Type    Data/Info
----------------------------
s          str     Hello world 
```

> **从脚本到函数**
> 
> 尽管仅使用脚本工作很诱人，即一个满是一个接一个命令的文件，但是要有计划的逐渐从脚本进化到一组函数：
> 
> *   脚本不可复用，函数可复用。
> 
> *   以函数的角度思考，有助于将问题拆分为小代码块。

### 1.1.3.3 IPython 提示与技巧

IPython 用户手册包含关于使用 IPython 的大量信息，但是，为了帮你你更快的入门，这里快速介绍三个有用的功能：*历史*，*魔法函数*，*别称*和*tab 完成*。

与 Unix Shell 相似，IPython 支持命令历史。按上下在之前输入的命令间切换：

```py
In [1]: x = 10
In [2]: <UP>
In [2]: x = 10 
```

IPython 通过在命令前加*%*字符的前缀，支持所谓魔法函数。例如，前面部分的函数*run*和*whos*都是魔法函数。请注意*automagic*设置默认是启用，允许你忽略前面的*%*。因此，你可以只输入魔法函数仍然是有效的。

其他有用的魔法函数：

*   **%cd** 改变当前目录

In [6]:

```py
cd .. 
```

```py
/Users/cloga/Documents 
```

*   **%timeit** 允许你使用来自标准库中的 timeit 模块来记录执行短代码端的运行时间

In [7]:

```py
timeit x = 10 
```

```py
10000000 loops, best of 3: 26.7 ns per loop 
```

*   **%cpaste** 允许你粘贴代码，特别是来自网站的代码，前面带有标准的 Python 提示符 (即 >>>) 或 ipython 提示符的代码(即 in [3])：

    ```py
    In [5]: cpaste
    Pasting code; enter ’--’ alone on the line to stop or use Ctrl-D. :In [3]: timeit x = 10
    :--
    10000000 loops, best of 3: 85.9 ns per loop
    In [6]: cpaste
    Pasting code; enter ’--’ alone on the line to stop or use Ctrl-D. :&gt;&gt;&gt; timeit x = 10
    :--
    10000000 loops, best of 3: 86 ns per loop 
    ```

*   **%debug** 允许你进入事后除错。也就是说，如果你想要运行的代码抛出了一个异常，使用**%debug**将在抛出异常的位置进入排错程序。

```py
In [7]: x === 10
File "<ipython-input-6-12fd421b5f28>", line 1
x === 10 ^
  SyntaxError: invalid syntax
In [8]: debug
> /home/esc/anaconda/lib/python2.7/site-packages/IPython/core/compilerop.py(87)ast_parse()
       86         and are passed to the built-in compile function."""
 ---> 87         return compile(source, filename, symbol, self.flags | PyCF_ONLY_AST, 1)
88
 ipdb>locals()
 {’source’: u’x === 10\n’, ’symbol’: ’exec’, ’self’:
 <IPython.core.compilerop.CachingCompiler instance at 0x2ad8ef0>,
 ’filename’: ’<ipython-input-6-12fd421b5f28>’} 
```

> **IPython help**
> 
> *   内置的 IPython 手册可以通过*%quickref*魔法函数进入。
> *   输入*%magic*会显示所有可用魔法函数的列表。

而且 IPython 提供了大量的*别称*来模拟常见的 UNIX 命令行工具比如*ls*等于 list files，*cp*等于 copy files 以及*rm*等于 remove files。输入*alias*可以显示所有的别称的列表：

In [5]:

```py
alias 
```

```py
Total number of aliases: 12 
```

Out[5]:

```py
[('cat', 'cat'),
 ('cp', 'cp'),
 ('ldir', 'ls -F -G -l %l | grep /$'),
 ('lf', 'ls -F -l -G %l | grep ^-'),
 ('lk', 'ls -F -l -G %l | grep ^l'),
 ('ll', 'ls -F -l -G'),
 ('ls', 'ls -F -G'),
 ('lx', 'ls -F -l -G %l | grep ^-..x'),
 ('mkdir', 'mkdir'),
 ('mv', 'mv'),
 ('rm', 'rm'),
 ('rmdir', 'rmdir')] 
```

最后，提一下*tab 完成*功能，我们从 IPython 手册引用它的描述：

> Tab completion, especially for attributes, is a convenient way to explore the structure of any object you’re dealing with. Simply type object_name. <tab>to view the object’s attributes. Besides Python objects and keywords, tab completion also works on file and directory names.</tab>

```py
In [1]: x = 10
In [2]: x.<TAB>
x.bit_length x.conjugate x.denominator x.imag x.numerator x.real
In [3]: x.real.
x.real.bit_length x.real.denominator x.real.numerator x.real.conjugate x.real.imag x.real.real
In [4]: x.real. 
```