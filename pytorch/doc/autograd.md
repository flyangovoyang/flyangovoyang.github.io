# PyTorch自动梯度机制

文本整理自PyTorch官方笔记[Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)，最近一次对齐时间为2023年3月。

本文将概述autograd是如何工作并记录运算的。

## autograd如何编码历史运算

autograd是一个反向自动微分系统。从概念上讲，autograd记录了一张有向无环图（计算图），其叶子结点是输入张量，根结点是输出张量，该图记录了数据的所有运算过程。通过从根结点到叶子结点的追踪，可以使用链式法则来自动计算梯度值。

autograd在内部用`Function`对象（实际上是表达式）来表示计算图，可以用`apply()`来获取计算图的结果。当进行前向传递时，autograd同时执行请求的计算，并构建一张图（用函数表示）来计算梯度（每个`torch.Tensor`的`grad_fn`属性是图的入口）。当前向传递计算完成后，通过反向传递来计算梯度。

需要注意的一点是，每轮迭代都会从头创建计算图，这正是PyTorch允许使用任意Python控制流语句来改变计算图整体形状和大小的原因。在启动训练之前，不需要对所有可能路径进行编码，实际执行的路径就是计算梯度的路径。

## 保存的张量

一些运算需要在前向传递过程中保存中间结果，以便执行反向传递。比如，函数$x \mapsto x^2$需要保存输入$x$来计算梯度。

当定义一个**自定义Python函数**时，可以在前向传递中使用`save_for_backward()`来保存张量，然后在反向传递时使用`saved_tensors`来获取它。更多信息可参考[Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html).

对于PyTorch定义的运算（例如`torch.pow()`），张量会根据需要自动保存。如果你想探索特定`grad_fn`存储了哪些张量（出于学习或者调试目的），可以查看其前缀为`_saved`开头的属性。

```python
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
print(x.equal(y.grad_fn._saved_self))  # True
print(x is y.grad_fn._saved_self)  # True
```

在上面的代码中，`y.grad_fn._saved_self`指向的就是张量`x`，但是情况并非总是如此，比如：

```python
x = torch.randn(5, requires_grad=True)
y = x.exp()
print(y.equal(y.grad_fn._saved_result))  # True
print(y is y.grad_fn._saved_result)  # False
```

为了避免引用循环，PyTorch在保存张量时将张量***打包（pack）***，然后***解包（unpack）***到不同的张量中以供读取。这里，`y.grad_fn._saved_result`不再是`y`，但是和`y`共享存储。

一个张量是否会被打包成不同的张量，取决于它是否是自己`grad_fn`的输出，这是一个存在多种情况的实现细节，用户不应该过于依赖。你可以借助[Hooks for saved tensors](https://pytorch.org/docs/stable/notes/autograd.html#saved-tensors-hooks-doc)来控制PyTorch怎样去打包和解包。

## 不可微函数的梯度

使用自动微分的梯度计算只在每个基础函数都可微的情况下有效，然而实际使用中很多函数都不具备这个属性，比如`relu`函数和在0处的`sqrt`函数。为了减少不可微函数的影响，按先后顺序执行如下规则来定义基础运算的梯度值：

1. 如果函数可微，梯度在当前点存在，则直接使用该梯度值；
2. 如果函数是凸的（至少局部是凸的），使用**最小范数的子梯度**（即梯度最陡峭的方向）；
3. 如果函数是凹的（至少局部是凹的），使用**最小范数的超梯度**（考虑`-f(x)`并使用之前的点）；
4. 如果函数已定义，借助连续性来定义当前点的梯度（注意梯度为`inf`也是可能的，比如`sqrt(0)`），如果梯度可能是多个值，随意选择一个）；
5. 如果函数没有定义（例如`sqrt(-1)`、`log(-1)`或者函数输入为`NaN`），梯度值是任意值（此时**可能**会抛出异常，但不保证）。绝大多数函数会使用`NaN`作为梯度，但是为了性能原因，一些函数可能会使用其他值（比如`log(-1)`）;
6. 如果函数不是一个确定性映射（比如不是一个数学函数），它会被标记为不可微，此时在梯度开启的情况下，将函数作用在需要`grad`的张量上，那么反向传播就会出错。

## 局部禁用梯度计算

局部禁用梯度计算的方法有两种，一种是利用`no-grad`上下文管理器（context manager）禁用整个代码块的梯度；另一种更为细粒度的控制方法是直接设置张量的`requires_grad`属性。

> 注意：`model.eval()`的作用是开启评测模式，它不会禁用梯度！下文会详细介绍eval()的功能。

### 设置requires_grad

`requires_grad`是一个布尔类型的标记，如果没有将张量包装进`nn.Parameter`，它的默认值是`False`. 这个属性可以用来细粒度地将部分网络层排除到梯度计算范围外。

在前向传递过程中，如果一个算子（operation）至少有一个输入张量的requires_grad为True，那么这个算子才会被记录到反向图中。在反向传递过程中，只有requires_grad=True的叶子结点的grad属性才会累计梯度值。

需要注意的是，**尽管每个张量都有requres_grad属性，但是设置该属性只对叶子结点张量有意义**。非叶子结点（有grad_fn属性）充当的作用是：因为他们的反向图关联了叶子结点，所以它们的梯度会作为中间结果来计算需要梯度的叶子结点的梯度。因而，所有非叶子结点的requires_grad会自动设置为True.

设置requires_grad应该是你控制模型哪部分参与梯度计算的主要方法。比如，如果你想在微调的时候，冻结预训练模型部分。

同样，也可以整体冻结module，比如`nn.Module.requires_grad_()`

### grad模式

除了设置requires_grad属性，PyTorch还提供三种模式来改变梯度计算范围，分别为：**grad模式**（默认模式）、**no-grad模式**和**推理模式**。三者通过上下文管理器和修饰器来切换。

- grad模式：如果没有指定no-grad模式和推理模式，默认情况下会开启grad模式。requires_grad=True只在grad模式下可以生效，在其余两个模式中requires_grad都会自动设置为False；
- no-grad模式：所有输入都关闭梯度；适用于禁用代码块的梯度。当torch.nn.init原地更新初始化参数的时候，会依赖no-grad模式来避免autograd追踪；
- 推理模式：推理模式是no-grad模式的极端版本，启动推理模式允许PyTorch进一步对模型进行加速。但是它也有一个弊端，在推理模式下创建的张量在退出推理模式后将无法用于autograd追踪的计算。更多信息可以参考[Inference Mode](https://pytorch.org/cppdocs/notes/inference_mode.html)

推理模式的使用有两种方法，一种是和no_grad一样开启一个上下文管理器，另一种是加函数修饰器。如下是推理模式的一个示例：

```python
>>> import torch
>>> x = torch.ones(1, 2, 3, requires_grad=True)
>>> with torch.inference_mode():
...   y = x * x
>>> y.requires_grad
False
>>> y._version
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
RuntimeError: Inference tensors do not track version counter.
>>> @torch.inference_mode()
... def func(x):
...   return x * x
>>> out = func(x)
>>> out.requires_grad
False
```

### 评估模式 nn.Module.eval()

评估模式不是局部禁用梯度的机制，考虑比较容易混淆，在这里做一个辨析。

从功能上看，module.eval()和module.train(False)等价，如果模型依赖于torch.nn.Dropout和torch.nn.BatchNorm2d，那么就需要根据实际情况调用model.eval()和model.train()，这样避免在验证集上的更新BatchNorm的统计信息。

建议无论模型是否包含训练模式特定的行为，都保持一个好习惯：在训练的时候使用model.train()，在评测或者测试的时候，使用model.eval()

## 使用autograd进行原地运算

在autograd使用原地运算是一件很复杂的事情，因此除非当前运算有严重的内存压力，在绝大多数场景都不推荐这种行为。