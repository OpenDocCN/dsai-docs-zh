# 2.1 Python 高级功能（Constructs）

**作者**: Zbigniew Jędrzejewski-Szmek

这一章是关于 Python 语言的高级特性-从不是每种语言都有这些特性的角度来说，也可以从他们在更复杂的程序和库中更有用这个角度来说，但是，并不是说特别专业或特别复杂。

需要强调的是本章是纯粹关于语言本身-关于由特殊语法支持的特性，用于补充 Python 标准库的功能，聪明的外部模块不会实现这部分特性。

开发 Python 程序语言的流程、语法是惟一的因为非常透明，提议的修改会在公共邮件列表中从多种角度去评估，最终的决策是来自于想象中的用例的重要性、带来更多语言特性所产生的负担、与其他语法的一致性及提议的变化是否易于读写和理解的权衡。这个流程被定型在 Python 增强建议中-[PEPs](http://www.python.org/dev/peps/)。因此，本章中的特性都是在显示出确实解决了现实问题，并且他们的使用尽可能简洁后才被添加的。

## 2.1.1 迭代器、生成器表达式和生成器

### 2.1.1.1 迭代器

简洁

重复的工作是浪费，用一个标准的特性替代不同的自产方法通常使事物的可读性和共用性更好。 Guido van Rossum — [为 Python 添加可选的静态输入](http://www.artima.com/weblogs/viewpost.jsp?thread=86641)

迭代器是继承了[迭代协议](http://docs.python.org/dev/library/stdtypes.html#iterator-types)的对象-本质上，这意味着它有一个[next](http://docs.python.org/2.7/library/stdtypes.html#iterator.next)方法，调用时会返回序列中的下一个项目，当没有东西可返回时，抛出[StopIteration](http://docs.python.org/2.7/library/exceptions.html#exceptions.StopIteration)异常。

迭代器对象只允许循环一次。它保留了单次迭代中的状态（位置），或者从另外的角度来看，序列上的每次循环需要一个迭代器对象。这意味着我们可以在一个序列上同时循环多次。从序列上分离出循环逻辑使我们可以有不止一种方法去循环。

在容器上调用[**iter**](http://docs.python.org/2.7/reference/datamodel.html#object.__iter__)方法来创建一个迭代器对象是获得迭代器的最简单方式。[iter](http://docs.python.org/2.7/library/functions.html#iter)函数帮我们完成这项工作，节省了一些按键次数。

In [12]:

```py
nums = [1,2,3]      # 注意 ... 变化: 这些是不同的对象
iter(nums) 
```

Out[12]:

```py
<listiterator at 0x105f8b490> 
```

In [2]:

```py
nums.__iter__() 
```

Out[2]:

```py
<listiterator at 0x105bd9bd0> 
```

In [3]:

```py
nums.__reversed__() 
```

Out[3]:

```py
<listreverseiterator at 0x105bd9c50> 
```

In [4]:

```py
it = iter(nums)
next(it)            # next(obj)是 obj.next()的简便用法 
```

Out[4]:

```py
1 
```

In [5]:

```py
it.next() 
```

Out[5]:

```py
2 
```

In [6]:

```py
next(it) 
```

Out[6]:

```py
3 
```

In [7]:

```py
next(it) 
```

```py
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-7-2cdb14c0d4d6> in <module>()
----> 1  next(it)

StopIteration: 
```

在一个循环上使用时，[StopIteration](http://docs.python.org/2.7/library/exceptions.html#exceptions.StopIteration) 被忍受了，使循环终止。但是，当显式调用时，我们可以看到一旦迭代器结束，再访问它会抛出异常。

使用[for..in](http://docs.python.org/2.7/reference/compound_stmts.html#for) 循环也使用`__iter__`方法。这个方法允许我们在序列上显式的开始一个循环。但是，如果我们已经有个迭代器，我们想要可以在一个循环中以同样的方式使用它。要做到这一点，迭代器及`next`也需要有一个称为`__iter__`的方法返回迭代器(`self`)。

Python 中对迭代器的支持是普遍的：标准库中的所有的序列和无序容器都支持迭代器。这个概念也被扩展到其他的事情：例如文件对象支持按行循环。

In [10]:

```py
f = open('./etc/fstab')
f is f.__iter__() 
```

Out[10]:

```py
True 
```

`文件`是迭代器本身，它的`__iter__`方法并不创建一个新的对象：仅允许一个单一线程的序列访问。

### 2.1.1.2 生成器表达式

创建迭代器对象的第二种方式是通过生成器表达式，这也是列表推导的基础。要增加明确性，生成器表达式通常必须被括号或表达式包围。如果使用圆括号，那么创建了一个生成器迭代器。如果使用方括号，那么过程被缩短了，我们得到了一个列表。

In [13]:

```py
(i for i in nums) 
```

Out[13]:

```py
<generator object <genexpr> at 0x105fbc320> 
```

In [14]:

```py
[i for i in nums] 
```

Out[14]:

```py
[1, 2, 3] 
```

In [15]:

```py
list(i for i in nums) 
```

Out[15]:

```py
[1, 2, 3] 
```

在 Python 2.7 和 3.x 中，列表推导语法被扩展为**字典和集合推导**。当生成器表达式被大括号包围时创建一个`集合`。当生成器表达式包含一对`键:值`的形式时创建`字典`:

In [16]:

```py
{i for i in range(3)} 
```

Out[16]:

```py
{0, 1, 2} 
```

In [17]:

```py
{i:i**2 for i in range(3)} 
```

Out[17]:

```py
{0: 0, 1: 1, 2: 4} 
```

如果你还在前面一些 Python 版本，那么语法只是有一点不同：

In [18]:

```py
set(i for i in 'abc') 
```

Out[18]:

```py
{'a', 'b', 'c'} 
```

In [19]:

```py
dict((i, ord(i)) for i in 'abc') 
```

Out[19]:

```py
{'a': 97, 'b': 98, 'c': 99} 
```

生成器表达式非常简单，在这里没有什么多说的。只有一个疑难问题需要提及：在旧的 Python 中，索引变量（i）可以泄漏，在>=3 以上的版本，这个问题被修正了。

### 2.1.1.3 生成器

生成器

生成器是一个可以产生一个结果序列而不是单一值的函数。

David Beazley — [协程和并发的有趣课程](http://www.dabeaz.com/coroutines/)

创建迭代器对应的第三种方法是调用生成器函数。**生成器**是包含关键字[yield](http://docs.python.org/2.7/reference/simple_stmts.html#yield)的函数。必须注意，只要这个关键词出现就会彻底改变函数的本质：这个`yield`关键字并不是必须激活或者甚至可到达，但是，会造成这个函数被标记为一个生成器。当普通函数被调用时，函数体内包含的指令就开始执行。当一个生成器被调用时，在函数体的第一条命令前停止执行。调用那个生成器函数创建一个生成器对象，继承迭代器协议。与调用普通函数一样，生成器也允许并发和递归。

当`next`被调用时，函数执行到第一个`yield`。每一次遇到`yield`语句都会给出`next`的一个返回值。执行完 yield 语句，就暂停函数的执行。

In [20]:

```py
def f():
    yield 1
    yield 2
f() 
```

Out[20]:

```py
<generator object f at 0x105fbc460> 
```

In [21]:

```py
gen = f()
gen.next() 
```

Out[21]:

```py
1 
```

In [22]:

```py
gen.next() 
```

Out[22]:

```py
2 
```

In [23]:

```py
gen.next() 
```

```py
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-23-b2c61ce5e131> in <module>()
----> 1  gen.next()

StopIteration: 
```

让我们进入一次调用生成器函数的生命周期。

In [24]:

```py
def f():
    print("-- start --")
    yield 3
    print("-- middle --")
    yield 4
    print("-- finished --")
gen = f()
next(gen) 
```

```py
-- start -- 
```

Out[24]:

```py
3 
```

In [25]:

```py
next(gen) 
```

```py
-- middle -- 
```

Out[25]:

```py
4 
```

In [26]:

```py
next(gen) 
```

```py
-- finished -- 
```

```py
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-26-67c2d9ac4268> in <module>()
----> 1  next(gen)

StopIteration: 
```

与普通函数不同，当执行`f()`时会立即执行第一个`print`，函数赋值到`gen`没有执行函数体内的任何语句。只有当用`next`激活`gen.next()`时，截至到第一个 yield 的语句才会被执行。第二个`next`打印`-- middle --`，执行到第二个`yield`终止。 第三个`next`打印`-- finished --`，并且到达了函数末尾。因为没有找到`yield`，抛出异常。

当向调用者传递控制时，在 yield 之后函数内发生了什么？每一个生成器的状态被存储在生成器对象中。从生成器函数的角度，看起来几乎是在一个独立的线程运行，但是，这是一个假象：执行是非常严格的单线程，但是解释器记录并恢复`next`值请求间的状态。

为什么生成器有用？正如迭代器部分的提到的，生成器只是创建迭代对象的不同方式。用`yield`语句可以完成的所有事，也都可以用`next`方法完成。尽管如此，使用函数，并让解释器执行它的魔法来创建迭代器有优势。函数比定义一个带有`next`和`__iter__`方法的类短很多。更重要的是，理解在本地变量中的状态比理解实例属性的状态对于生成器的作者来说要简单的多，对于后者来事必须要在迭代对象上不断调用`next`。

更广泛的问题是为什么迭代器有用？当迭代器被用于循环时，循环变的非常简单。初始化状态、决定循环是否结束以及寻找下一个值的代码被抽取到一个独立的的地方。这强调了循环体 - 有趣的部分。另外，这使在其他地方重用这些迭代体成为可能。

### 2.1.1.4 双向沟通

每个`yield`语句将一个值传递给调用者。这是由[PEP 255](http://www.python.org/dev/peps/pep-0255)（在 Python2.2 中实现）引入生成器简介的原因。但是，相反方向的沟通也是有用的。一个明显的方式可以是一些外部状态，全局变量或者是共享的可变对象。感谢[PEP 342](http://www.python.org/dev/peps/pep-0342)（在 2.5 中实现）使直接沟通成为可能。它通过将之前枯燥的`yeild`语句转换为表达式来实现。当生成器在一个`yeild`语句后恢复执行，调用者可以在生成器对象上调用一个方法，或者向生成器内部传递一个值，稍后由`yield`语句返回，或者一个不同的方法向生成器注入一个异常。

第一个新方法是[send(value)](http://docs.python.org/2.7/reference/expressions.html#generator.send)，与[next()](http://docs.python.org/2.7/reference/expressions.html#generator.next)类似，但是，向生成器传递值用于`yield`表达式来使用。实际上，`g.next()`和`g.send(None)`是等价的。

第二个新方法是[throw(type, value=None, traceback=None)](http://docs.python.org/2.7/reference/expressions.html#generator.throw)等价于：

In [ ]:

```py
raise type, value, traceback 
```

在`yield`语句的点上。

与[raise](https://docs.python.org/2.7/reference/simple_stmts.html#raise)不同 (在当前执行的点立即抛出异常), `throw()`只是首先暂停生成器，然后抛出异常。挑选 throw 这个词是因为它让人联想到将异常放在不同的位置，这与其他语言中的异常相似。

当异常在生成器内部抛出时发生了什么？它可以是显性抛出或者当执行一些语句时，或者它可以注入在`yield`语句的点上，通过`throw()`方法的意思。在任何情况下，这些异常用一种标准方式传播：它可以被`except`或`finally`语句监听，或者在其他情况下，它引起生成器函数的执行中止，并且传播给调用者。

为了完整起见，应该提一下生成器迭代器也有[close()](http://docs.python.org/2.7/reference/expressions.html#generator.close)函数，可以用来强制一个可能在其他情况下提供更多的值的生成器立即结束。它允许生成器[**del**](http://docs.python.org/2.7/reference/datamodel.html#object.__del__)函数去销毁保持生成器状态的对象。

让我们定义一个生成器，打印通过 send 和 throw 传递的内容。

In [2]:

```py
import itertools
def g():
    print '--start--'
    for i in itertools.count():
        print '--yielding %i--' % i
        try:
            ans = yield i
        except GeneratorExit:
            print '--closing--'
            raise
        except Exception as e:
             print '--yield raised %r--' % e
        else:
             print '--yield returned %s--' % ans 
```

In [3]:

```py
it = g()
next(it) 
```

```py
--start--
--yielding 0-- 
```

Out[3]:

```py
0 
```

In [4]:

```py
it.send(11) 
```

```py
--yield returned 11--
--yielding 1-- 
```

Out[4]:

```py
1 
```

In [5]:

```py
it.throw(IndexError) 
```

```py
--yield raised IndexError()--
--yielding 2-- 
```

Out[5]:

```py
2 
```

In [6]:

```py
it.close() 
```

```py
--closing-- 
```

**next 还是 **next**?**

在 Python2.X 中，迭代器用于取回下一个值的方法是调用[next](http://docs.python.org/2.7/library/stdtypes.html#iterator.next)。它通过全局方法[next](http://docs.python.org/2.7/library/stdtypes.html#iterator.next)来唤醒，这意味着它应该调用**next**。就像全局函数[iter](http://docs.python.org/2.7/library/functions.html#iter)调用**iter**。在 Python 3.X 中修正了这种前后矛盾，it.next 变成 it.**next**。对于其他的生成器方法 - `send`和`throw`-情况更加复杂，因为解释器并不隐性的调用它们。尽管如此，人们提出一种语法扩展，以便允许`continue`接收一个参数，用于传递给循环的迭代器的[send](http://docs.python.org/2.7/reference/expressions.html#generator.send)。如果这个语法扩展被接受，那么可能`gen.send`将变成`gen.__send__`。最后一个生成器函数，[close](http://docs.python.org/2.7/reference/expressions.html#generator.close)非常明显是命名错误，因为，它已经隐性被唤起。

### 2.1.1.5 生成器链

**注**：这是[PEP 380](http://www.python.org/dev/peps/pep-0380)的预览（没有实现，但是已经被 Python3.3 接受）。

假设我们正在写一个生成器，并且我们想要量产（yield）由第二个生成器生成的一堆值，**子生成器**。如果只关心量产值，那么就可以没任何难度的用循环实现，比如

In [ ]:

```py
for v in subgen:
    yield v 
```

但是，如果子生成器想要与调用者通过`send()`、`throw()`和`close()`正确交互，事情就会变得复杂起来。`yield`语句必须用[try..except..finally](http://docs.python.org/2.7/reference/compound_stmts.html#try)结构保护起来，与前面的生成器函数“degug”部分定义的类似。在[PEP 380](http://www.python.org/dev/peps/pep-0380#id13)提供了这些代码，现在可以说在 Python 3.3 中引入的新语法可以适当的从子生成器量产：

In [ ]:

```py
yield from some_other_generator() 
```

这个行为与上面的显性循环类似，重复从`some_other_generator`量产值直到生成器最后，但是，也可以向前对子生成器`send`、`throw`和`close`。

## 2.1.2 修饰器

概述

这个令人惊讶功能在这门语言中出现几乎是有歉意的，并且担心它是否真的那么有用。

Bruce Eckel — Python 修饰器简介

因为函数或类是对象，因此他们都可以传递。因为可以是可变的对象，所以他们可以被修改。函数或类对象被构建后，但是在绑定到他们的名称之前的修改行为被称为修饰。

在“修饰器”这个名称后面隐藏了两件事-一件是进行修饰工作（即进行真实的工作）的函数，另一件是遵守修饰器语法的表达式，[[email protected]](cdn-cgi/l/email-protection)

用函数的修饰器语法可以修饰函数：

In [ ]:

```py
@decorator             # ②
def function():        # ①
    pass 
```

*   用标准形式定义的函数。①
*   [[email protected]](cdn-cgi/l/email-protection)��[[email protected]](cdn-cgi/l/email-protection)，通常，这只是函数或类的名字。这部分首先被评估，在下面的函数定义完成后，修饰器被调用，同时将新定义的函数对象作为惟一的参数。修饰器的返回值被附加到函数的原始名称上。

修饰器可以被应用于函数和类。对于类，语法是一样的 - 原始类定义被作为一个参数来调用修饰器，并且无论返回什么都被赋给原始的名称。在修饰器语法实现之前（[PEP 318](http://www.python.org/dev/peps/pep-0318)），通过将函数或类对象赋给一个临时的变量，然后显性引用修饰器，然后将返回值赋给函数的名称，也可以到达相同的效果。这听起来像是打更多的字，确实是这样，并且被修饰函数的名字也被打了两次，因为临时变量必须被使用至少三次，这很容易出错。无论如何，上面的例子等同于：

In [ ]:

```py
def function():                  # ①
    pass
function = decorator(function)   # ② 
```

修饰器可以嵌套 - 应用的顺序是由底到顶或者由内到外。含义是最初定义的函数被第一个修饰器作为参数使用，第一个修饰器返回的内容被用于第二个修饰器的参数，...，最后一个修饰器返回的内容被绑定在最初的函数名称下。

选择这种修饰器语法是因为它的可读性。因为是在函数头之前指定的，很明显它并不是函数体的一部分，并且很显然它只能在整个函数上运行。因为，[[email protected]](cdn-cgi/l/email-protection)（"在你脸上"，按照 PEP 的说法 :)）。当使用多个修饰器时，每一个都是单独的一行，一种很容易阅读的方式。

### 2.1.2.1 替换或调整原始对象

修饰器可以返回相同的函数或类对象，也可以返回完全不同的对象。在第一种情况下，修饰器可以利用函数和类对象是可变的这个事实，并且添加属性，即为类添加修饰字符串。修饰器可以做一些有用的事甚至都不需要修改对象，例如，在全局登记中登记被修饰的类。在第二种情况下，虚拟任何东西都是可能的：当原始函数或类的一些东西被替换了，那么新对象就可以是完全不同的。尽管如此，这种行为不是修饰器的目的：他们的目的是微调被修饰的对象，而不是一些不可预测的东西。因此，当一个”被修饰的“函数被用一个不同的函数替换，新函数通常调用原始的函数，在做完一些预备工作之后。同样的，当”被修饰的“类被新的类替换，新类通常也来自原始类。让修饰器的目的是”每次“都做一些事情，比如在修饰器函数中登记每次调用，只能使用第二类修饰器。反过来，如果第一类就足够了，那么最好使用第一类，因为，它更简单。

### 2.1.2.2 像类和函数一样实现修饰器

修饰器的惟一一个要求是可以用一个参数调用。这意味着修饰器可以像一般函数一样实现，或者像类用**call**方法实现，或者在理论上，甚至是 lambda 函数。让我们比较一下函数和类的方法。修饰器表达式（@后面的部分）可以仅仅是一个名字，或者一次调用。仅使用名字的方式很好（输入少，看起来更整洁等），但是，只能在不需要参数来自定义修饰器时使用。作为函数的修饰器可以用于下列两个情况：

In [1]:

```py
def simple_decorator(function):
    print "doing decoration"
    return function
@simple_decorator
def function():
    print "inside function" 
```

```py
doing decoration 
```

In [2]:

```py
function() 
```

```py
inside function 
```

In [6]:

```py
def decorator_with_arguments(arg):
    print "defining the decorator"
    def _decorator(function):
        # in this inner function, arg is available too
        print "doing decoration,", arg
        return function
    return _decorator

@decorator_with_arguments("abc")
def function():
    print "inside function" 
```

```py
defining the decorator
doing decoration, abc 
```

上面两个修饰器属于返回原始函数的修饰器。如果他们返回一个新的函数，则需要更多一层的嵌套。在最坏的情况下，三层嵌套的函数。

In [7]:

```py
def replacing_decorator_with_args(arg):
    print "defining the decorator"
    def _decorator(function):
        # in this inner function, arg is available too
        print "doing decoration,", arg
        def _wrapper(*args, **kwargs):
            print "inside wrapper,", args, kwargs
            return function(*args, **kwargs)
        return _wrapper
    return _decorator
@replacing_decorator_with_args("abc")
def function(*args, **kwargs):
    print "inside function,", args, kwargs
    return 14 
```

```py
defining the decorator
doing decoration, abc 
```

In [8]:

```py
function(11, 12) 
```

```py
inside wrapper, (11, 12) {}
inside function, (11, 12) {} 
```

Out[8]:

```py
14 
```

定义`_wrapper`函数来接收所有位置和关键词参数。通常，我们并不知道被修饰的函数可能接收什么参数，因此封装器函数只是向被封装的函数传递所有东西。一个不幸的结果是有误导性的表面函数列表。

与定义为函数的修饰器相比，定义为类的复杂修饰器更加简单。当一个对象创建后，**init**方法仅允许返回`None`，已创建的对象类型是不可以修改的。这意味着当一个被作为类创建后，因此使用少参模式没有意义：最终被修饰的对象只会是由构建器调用返回的修饰对象的一个实例，并不是十分有用。因此，只需要探讨在修饰器表达式中带有参数并且修饰器**init**方法被用于修饰器构建，基于类的修饰器。

In [9]:

```py
class decorator_class(object):
    def __init__(self, arg):
        # this method is called in the decorator expression
        print "in decorator init,", arg
        self.arg = arg
    def __call__(self, function):
        # this method is called to do the job
        print "in decorator call,", self.arg
        return function 
```

In [10]:

```py
deco_instance = decorator_class('foo') 
```

```py
in decorator init, foo 
```

In [11]:

```py
@deco_instance
def function(*args, **kwargs):
    print "in function,", args, kwargs 
```

```py
in decorator call, foo 
```

In [12]:

```py
function() 
```

```py
in function, () {} 
```

与通用规则相比（[PEP 8](http://www.python.org/dev/peps/pep-0008)），将修饰器写为类的行为更像是函数，因此，他们的名字通常是以小写字母开头。

在现实中，创建一个新类只有一个返回原始函数的修饰器是没有意义的。人们认为对象可以保留状态，当修饰器返回新的对象时，这个修饰器更加有用。

In [13]:

```py
class replacing_decorator_class(object):
    def __init__(self, arg):
        # this method is called in the decorator expression
        print "in decorator init,", arg
        self.arg = arg
    def __call__(self, function):
        # this method is called to do the job
        print "in decorator call,", self.arg
        self.function = function
        return self._wrapper
    def _wrapper(self, *args, **kwargs):
        print "in the wrapper,", args, kwargs
        return self.function(*args, **kwargs) 
```

In [14]:

```py
deco_instance = replacing_decorator_class('foo') 
```

```py
in decorator init, foo 
```

In [15]:

```py
@deco_instance
def function(*args, **kwargs):
    print "in function,", args, kwargs 
```

```py
in decorator call, foo 
```

In [16]:

```py
function(11, 12) 
```

```py
in the wrapper, (11, 12) {}
in function, (11, 12) {} 
```

像这样一个修饰器可以非常漂亮的做任何事，因为它可以修改原始的函数对象和参数，调用或不调用原始函数，向后修改返回值。

### 2.1.2.3 复制原始函数的文档字符串和其他属性

当修饰器返回一个新的函数来替代原始的函数时，一个不好的结果是原始的函数名、原始的文档字符串和原始参数列表都丢失了。通过设置**doc**（文档字符串）、**module**和**name**（完整的函数），以及**annotations**（关于参数和返回值的额外信息，在 Python 中可用）可以部分”移植“这些原始函数的属性到新函数的设定。这可以通过使用[functools.update_wrapper](http://docs.python.org/2.7/library/functools.html#functools.update_wrapper)来自动完成。

In [ ]:In [17]:

```py
import functools
def better_replacing_decorator_with_args(arg):
    print "defining the decorator"
    def _decorator(function):
        print "doing decoration,", arg
        def _wrapper(*args, **kwargs):
            print "inside wrapper,", args, kwargs
            return function(*args, **kwargs)
        return functools.update_wrapper(_wrapper, function)
    return _decorator
@better_replacing_decorator_with_args("abc")
def function():
    "extensive documentation"
    print "inside function"
    return 14 
```

```py
defining the decorator
doing decoration, abc 
```

In [18]:

```py
function 
```

Out[18]:

```py
<function __main__.function> 
```

In [19]:

```py
print function.__doc__ 
```

```py
extensive documentation 
```

在属性列表中缺少了一个重要的东西：参数列表，这些属性可以复制到替换的函数。参数的默认值可以用`__defaults__`、`__kwdefaults__`属性来修改，但是，不幸的是参数列表本身不能设置为属性。这意味着`help(function)`将显示无用的参数列表，对于函数用户造成困扰。一种绕过这个问题的有效但丑陋的方法是使用`eval`来动态创建一个封装器。使用外部的`decorator`模块可以自动完成这个过程。它提供了对`decorator`装饰器的支持，给定一个封装器将它转变成保留函数签名的装饰器。

总结一下，装饰器通常应该用`functools.update_wrapper`或其他方式来复制函数属性。

### 2.1.2.4 标准类库中的实例

首先，应该说明，在标准类库中有一些有用的修饰器。有三类装饰器确实构成了语言的一部分：

*   [classmethod](http://docs.python.org/2.7/library/functions.html#classmethod) 造成函数成为“类方法”，这意味着不需要创建类的实例就可以激活它。当普通的方法被激活后，解释器将插入一个实例对象作为第一个位置参数，`self`。当类方法被激活后，类自身被作为一点参数，通常称为`cls`。

    类方法仍然可以通过类的命名空间访问，因此，他们不会污染模块的命名空间。类方法可以用来提供替代的构建器：

In [1]:

```py
class Array(object):
    def __init__(self, data):
        self.data = data

    @classmethod
    def fromfile(cls, file):
        data = numpy.load(file)
        return cls(data) 
```

```py
这是一个清洁器，然后使用大量的标记来`__init__`。 
```

*   [staticmethod](http://docs.python.org/2.7/library/functions.html#staticmethod)用来让方法“静态”，即，从根本上只一个普通的函数，但是可以通过类的命名空间访问。当函数只在这个类的内部需要时（它的名字应该与 _ 为前缀），或者当我们想要用户认为方法是与类关联的，尽管实施并不需要这样。

*   [property](http://docs.python.org/2.7/library/functions.html#property)是对 getters 和 setters pythonic 的答案。用`property`修饰过的方法变成了一个 getter，getter 会在访问属性时自动调用。

In [2]:

```py
class A(object):
    @property
    def a(self):
        "an important attribute"
        return "a value" 
```

In [3]:

```py
A.a 
```

Out[3]:

```py
<property at 0x104139260> 
```

In [4]:

```py
A().a 
```

Out[4]:

```py
'a value' 
```

在这个例子中，`A.a`是只读的属性。它也写入了文档：`help(A)`包含从 getter 方法中拿过来的属性的文档字符串。将`a`定义为一个属性允许实时计算，副作用是它变成只读，因为没有定义 setter。

要有 setter 和 getter，显然需要两个方法。从 Python 2.6 开始，下列语法更受欢迎：

In [5]:

```py
class Rectangle(object):
    def __init__(self, edge):
        self.edge = edge

    @property
    def area(self):
        """Computed area.

 Setting this updates the edge length to the proper value.
 """
        return self.edge**2

    @area.setter
    def area(self, area):
        self.edge = area ** 0.5 
```

这种方式有效是因为`property`修饰器用 property 对象替换 getter 方法。这个对象反过来有三个方法，`getter`、`setter`和`deleter`，可以作为修饰器。他们的任务是设置 property 对象的 getter、 setter 和 deleter（存储为`fget`、`fset`和`fdel`属性）。当创建一个对象时，getter 可以像上面的例子中进行设置。当定义一个 setter，我们已经在`area`下有 property 对象，我们通过使用 setter 方法为它添加 setter。所有的这些发生在我们创建类时。

接下来，当类的实例被创建后，property 对象是特别的，当解释器执行属性访问，属性赋值或者属性删除时，任务被委托给 property 对象的方法。

为了让每个事情都清晰，让我们定义一个“debug”例子：

In [6]:

```py
class D(object):
    @property
    def a(self):
        print "getting", 1
        return 1
    @a.setter
    def a(self, value):
        print "setting", value
    @a.deleter
    def a(self):
        print "deleting" 
```

In [7]:

```py
D.a 
```

Out[7]:

```py
<property at 0x104139520> 
```

In [8]:

```py
D.a.fget 
```

Out[8]:

```py
<function __main__.a> 
```

In [9]:

```py
D.a.fset 
```

Out[9]:

```py
<function __main__.a> 
```

In [10]:

```py
D.a.fdel 
```

Out[10]:

```py
<function __main__.a> 
```

In [12]:

```py
d = D()               # ... varies, this is not the same `a` function
d.a 
```

```py
getting 1 
```

Out[12]:

```py
1 
```

In [13]:

```py
d.a = 2 
```

```py
setting 2 
```

In [14]:

```py
del d.a 
```

```py
deleting 
```

In [15]:

```py
d.a 
```

```py
getting 1 
```

Out[15]:

```py
1 
```

属性是修饰语语法的极大扩展。修饰器语法的一个前提-名字不可以重复-被违背了，但是，到目前位置没有什么事变糟了。为 getter、setter 和 deleter 方法使用相同的名字是一个好风格。

一些更新的例子包括：

*   functools.lru_cache 记忆任意一个函数保持有限的 arguments\:answer 对缓存（Python 3.2）
*   [functools.total_ordering](http://docs.python.org/2.7/library/functools.html#functools.total_ordering)是一类修饰器，根据单一的可用方法（Python 2.7）补充缺失的顺序方法（**lt**, **gt**, **le**, ...）。

### 2.1.2.5 函数废弃

假如我们想要在我们不再喜欢的函数第一次激活时在`stderr`打印废弃警告。如果我们不像修改函数，那么我们可以使用修饰器：

In [16]:

```py
class deprecated(object):
    """Print a deprecation warning once on first use of the function.

 >>> @deprecated()                    # doctest: +SKIP
 ... def f():
 ...     pass
 >>> f()                              # doctest: +SKIP
 f is deprecated
 """
    def __call__(self, func):
        self.func = func
        self.count = 0
        return self._wrapper
    def _wrapper(self, *args, **kwargs):
        self.count += 1
        if self.count == 1:
            print self.func.__name__, 'is deprecated'
        return self.func(*args, **kwargs) 
```

也可以将其实施为一个函数：

In [17]:

```py
def deprecated(func):
    """Print a deprecation warning once on first use of the function.

 >>> @deprecated                      # doctest: +SKIP
 ... def f():
 ...     pass
 >>> f()                              # doctest: +SKIP
 f is deprecated
 """
    count = [0]
    def wrapper(*args, **kwargs):
        count[0] += 1
        if count[0] == 1:
            print func.__name__, 'is deprecated'
        return func(*args, **kwargs)
    return wrapper 
```

### 2.1.2.6 A while-loop 删除修饰器

假如我们有一个函数返回事物列表，这个列表由循环创建。如果我们不知道需要多少对象，那么这么做的标准方式是像这样的：

In [18]:

```py
def find_answers():
    answers = []
    while True:
        ans = look_for_next_answer()
        if ans is None:
            break
        answers.append(ans)
    return answers 
```

只要循环体足够紧凑，这是可以的。一旦循环体变得更加负责，就像在真实代码中，这种方法的可读性将很差。我们可以通过使用 yield 语句来简化，不过，这样的话，用户需要显性的调用列表（find_answers()）。

我们可以定义一个修饰器来为我们构建修饰器：

In [19]:

```py
def vectorized(generator_func):
    def wrapper(*args, **kwargs):
        return list(generator_func(*args, **kwargs))
    return functools.update_wrapper(wrapper, generator_func) 
```

接下来我们的函数变成：

In [ ]:

```py
@vectorized
def find_answers():
    while True:
        ans = look_for_next_answer()
        if ans is None:
            break
        yield ans 
```

### 2.1.2.7 插件注册系统

这是一个不会修改类的类修饰器，但是，只要将它放在全局注册域。它会掉入返回原始对象的修饰器类别中：

In [21]:

```py
class WordProcessor(object):
    PLUGINS = []
    def process(self, text):
        for plugin in self.PLUGINS:
            text = plugin().cleanup(text)
        return text

    @classmethod
    def plugin(cls, plugin):
        cls.PLUGINS.append(plugin)

@WordProcessor.plugin
class CleanMdashesExtension(object):
    def cleanup(self, text):
        return text.replace('&mdash;', u'\N{em dash}') 
```

这里我们用修饰器来分权插件注册。修饰器是名词而不是动词，因为我们用它来声明我们的类是`WordProcessor`的一个插件。方法`plugin`只是将类添加到插件列表中。

关于这个插件本身多说一句：它用实际的 Unicode 的 em-dash 字符替换了 em-dash HTML 实体。它利用[unicode 绝对标记](http://docs.python.org/2.7/reference/lexical_analysis.html#string-literals)来通过字符在 unicode 数据库（“EM DASH”）中的名字来插入字符。如果直接插入 Unicode 字符，将无法从程序源文件中区分 en-dash。

### 2.1.2.8 更多例子和阅读

*   [PEP 318](http://www.python.org/dev/peps/pep-0318)（函数和方法的修饰器语法）
*   [PEP 3129](http://www.python.org/dev/peps/pep-3129)（类修饰器语法）
*   [`wiki.python.org/moin/PythonDecoratorLibrary`](http://wiki.python.org/moin/PythonDecoratorLibrary)
*   [`docs.python.org/dev/library/functools.html`](http://docs.python.org/dev/library/functools.html)
*   [`pypi.python.org/pypi/decorator`](http://pypi.python.org/pypi/decorator)
*   Bruce Eckel
    *   Decorators I: Introduction to Python Decorators
    *   Python Decorators II: Decorator Arguments
    *   Python Decorators III: A Decorator-Based Build System

## 2.1.3 上下文管理器

上下文管理器是带有`__enter__`和`__exit__`方法的对象，在 with 语句中使用：

In [ ]:

```py
with manager as var:
    do_something(var) 
```

最简单的等价 case 是

In [ ]:

```py
var = manager.__enter__()
try:
    do_something(var)
finally:
    manager.__exit__() 
```

换句话说，在[PEP343](http://www.python.org/dev/peps/pep-0343)定义的上下文管理器协议，使将[try..except..finally](http://docs.python.org/2.7/reference/compound_stmts.html#try)结构中枯燥的部分抽象成一个独立的类，而只保留有趣的`do_something`代码块成为可能。

1.  首先调用[**enter**](http://docs.python.org/2.7/reference/datamodel.html#object.__enter__)方法。它会返回一个值被赋值给`var`。`as`部分是可选的：如果不存在，`__enter__`返回的值将被忽略。
2.  `with`下面的代码段将被执行。就像`try`从句一样，它要么成功执行到最后，要么[break](http://docs.python.org/2.7/reference/simple_stmts.html#break)、[continue](http://docs.python.org/2.7/reference/simple_stmts.html#continue)或者[return](http://docs.python.org/2.7/reference/simple_stmts.html#return)，或者它抛出一个异常。无论哪种方式，在这段代码结束后，都将调用[**exit**](http://docs.python.org/2.7/reference/datamodel.html#object.__exit__)。如果抛出异常，关于异常的信息会传递给`__exit__`，将在下一个部分描述。在一般的情况下，异常将被忽略，就像`finally`从句一样，并且将在`__exit__`结束时重新抛出。

假如我们想要确认一下文件是否在我们写入后马上关闭：

In [23]:

```py
class closing(object):
    def __init__(self, obj):
        self.obj = obj
    def __enter__(self):
        return self.obj
    def __exit__(self, *args):
        self.obj.close()
with closing(open('/tmp/file', 'w')) as f:
    f.write('the contents\n') 
```

这里我们确保当`with`代码段退出后，`f.close()`被调用。因为关闭文件是非常常见的操作，对这个的支持已经可以在`file`类中出现。它有一个**exit**方法，调用了`close`并且被自己用于上下文管理器：

In [ ]:

```py
with open('/tmp/file', 'a') as f:
    f.write('more contents\n') 
```

`try..finally`的常用用途是释放资源。不同的情况都是类似的实现：在`__enter__`阶段，是需要资源的，在`__exit__`阶段，资源被释放，并且异常，如果抛出的话，将被传递。就像 with 文件一样，当一个对象被使用后通常有一些自然的操作，最方便的方式是由一个内建的支持。在每一次发布中，Python 都在更多的地方提供了支持：

*   所以类似文件的对象：
    *   [file](http://docs.python.org/2.7/library/functions.html#file) ➔ 自动关闭
    *   [fileinput](http://docs.python.org/2.7/library/fileinput.html#fileinput)，[tempfile](http://docs.python.org/2.7/library/tempfile.html#tempfile) (py >= 3.2)
    *   [bz2.BZ2File](http://docs.python.org/2.7/library/bz2.html#bz2.BZ2File)，[gzip.GzipFile](http://docs.python.org/2.7/library/gzip.html#gzip.GzipFile)，[tarfile.TarFile](http://docs.python.org/2.7/library/tarfile.html#tarfile.TarFile)，[zipfile.ZipFile](http://docs.python.org/2.7/library/zipfile.html#zipfile.ZipFile)
    *   [ftplib](http://docs.python.org/2.7/library/ftplib.html#ftplib)，[nntplib](http://docs.python.org/2.7/library/nntplib.html#nntplib) ➔ 关闭连接 (py >= 3.2 或 3.3)
*   锁
    *   [multiprocessing.RLock](http://docs.python.org/2.7/library/multiprocessing.html#multiprocessing.RLock) ➔ 锁和解锁
    *   [multiprocessing.Semaphore](http://docs.python.org/2.7/library/multiprocessing.html#multiprocessing.Semaphore)
    *   [memoryview](http://docs.python.org/2.7/library/stdtypes.html#memoryview) ➔ 自动释放 (py >= 3.2 和 2.7)
*   [decimal.localcontext](http://docs.python.org/2.7/library/decimal.html#decimal.localcontext) ➔ 临时修改计算的精度
*   _winreg.PyHKEY ➔ 打开或关闭 hive 键
*   warnings.catch_warnings ➔ 临时杀掉警告
*   contextlib.closing ➔ 与上面的例子类似，调用`close`
*   并行程序
    *   concurrent.futures.ThreadPoolExecutor ➔ 激活并行，然后杀掉线程池 (py >= 3.2)
    *   concurrent.futures.ProcessPoolExecutor ➔ 激活并行，然后杀掉进程池 (py >= 3.2)
    *   nogil ➔ 临时解决 GIL 问题 (仅 cython :( )

### 2.1.3.1 捕捉异常

当`with`代码块中抛出了异常，异常会作为参数传递给`__exit__`。与[sys.exc_info()](http://docs.python.org/2.7/library/sys.html#sys.exc_info)类似使用三个参数：type, value, traceback。当没有异常抛出时，`None`被用于三个参数。上下文管理器可以通过从`__exit__`返回 true 值来“吞下”异常。可以很简单的忽略异常，因为如果`__exit__`没有使用`return`，并且直接运行到最后，返回`None`，一个 false 值，因此，异常在`__exit__`完成后重新抛出。

捕捉异常的能力开启了一些有趣的可能性。一个经典的例子来自于单元测试-我们想要确保一些代码抛出正确类型的异常：

In [2]:

```py
class assert_raises(object):
    # based on pytest and unittest.TestCase
    def __init__(self, type):
        self.type = type
    def __enter__(self):
        pass
    def __exit__(self, type, value, traceback):
        if type is None:
            raise AssertionError('exception expected')
        if issubclass(type, self.type):
            return True # swallow the expected exception
        raise AssertionError('wrong exception type')

with assert_raises(KeyError):
    {}['foo'] 
```

### 2.1.3.2 使用生成器定义上下文管理器

当讨论生成器时，曾说过与循环相比，我们更偏好将生成器实现为一个类，因为，他们更短、更美妙，状态存储在本地，而不是实例和变量。另一方面，就如在双向沟通中描述的，生成器和它的调用者之间的数据流动可以是双向的。这包含异常，可以在生成器中抛出。我们希望将上下文生成器实现为一个特殊的生成器函数。实际上，生成器协议被设计成可以支持这个用例。

In [ ]:

```py
@contextlib.contextmanager
def some_generator(<arguments>):
    <setup>
    try:
        yield <value>
    finally:
        <cleanup> 
```

[contextlib.contextmanager](http://docs.python.org/2.7/library/contextlib.html#contextlib.contextmanager)帮助者可以将一个生成器转化为上下文管理器。生成器需要遵循一些封装器函数强加的规则--它必须`yield`一次。在`yield`之前的部分是从`__enter__`来执行，当生成器在`yield`挂起时，由上下文管理器保护的代码块执行。如果抛出异常，解释器通过`__exit__`参数将它交给封装器，然后封装器函数在`yield`语句的点抛出异常。通过使用生成器，上下文管理器更短和简单。

让我们将`closing`例子重写为一个生成器：

In [ ]:

```py
@contextlib.contextmanager
def closing(obj):
    try:
        yield obj
    finally:
        obj.close() 
```

让我们将`assert_raises`例子重写为生成器：

In [ ]:

```py
@contextlib.contextmanager
def assert_raises(type):
    try:
        yield
    except type:
        return
    except Exception as value:
        raise AssertionError('wrong exception type')
    else:
        raise AssertionError('exception expected') 
```

这里我们使用修饰器来将一个生成器函数转化为上下文管理器！

In [1]:

```py
%matplotlib inline
import numpy as np 
```