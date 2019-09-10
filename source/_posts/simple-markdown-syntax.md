---
title: 简明Markdown语法
mathjax: true
toc: false
comments: true
date: 2019-09-08 08:40:16
categories: markdown
tags: markdown
---

<a name='inner-link' id='你看不见我，哈哈哈~'></a>

> 工欲善其事，必先利其器

既然开始专心写博客，自然就需要对markdown的语法了然于胸。之前虽然也用过一些特性，但是，总归不能有一种驾轻就熟的感觉。作为博客的开篇之作，今天就好好说道说道。（一个有意思的事情是，这篇博客也是用markdown语言写的）

<!--more-->

目前市面上对markdown的实现有很多版本，本着从简去繁、通用高效的原则，经过多方对比，最终选择了Github官方的markdown实现版本。[Github官方语法规范](https://help.github.com/en/articles/basic-writing-and-formatting-syntax)

话不多说，上菜！

# 1. 标题

```text
# 一级标题  
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

# 一级标题

## 二级标题

### 三级标题

#### 四级标题

##### 五级标题

###### 六级标题

# 2. 文本风格

```text
**粗体风格**

*斜体风格*

~~删除线风格~~

**粗体嵌套 _斜体_ 风格**

***又粗又斜体风格***
```

**粗体风格**

*斜体风格*

~~删除线风格~~

**粗体嵌套 _斜体_ 风格**

***又粗又斜体风格***

# 3. 文本引用

```text
文本引用内也可以加入其它markdown语法

> ## 名人名言
> 这篇文章写的真他妈透彻。 —— 鲁迅
> > 上面的不是我说的，但是我+1。 —— 鲁迅
```

> ## 名人名言
> 这篇文章写的真他妈透彻。 —— 鲁迅
> > 上面的不是我说的，但是我+1。 —— 鲁迅

# 4. 代码引用

```text
下面用的都是反引号（键盘左上角，esc下面那个）

shell命令示例：`rm -rf /`

为了不引起解析错误，下面用'代替了`(无奈之举)
'''python
import os

print("代码块指定语言，可以高亮显示哦")
'''
```

shell命令示例：`rm -rf /`

```python
import os

print("代码块指定语言，可以高亮显示哦")
```

# 5. 链接

```text
让我们屏住呼吸，[一起回到最初的起点](#inner-link)。这个需要在页面指定的位置定义一个锚点，类似于这样`<a name='inner-line'></a>`

<https://ritchie.top>

<god@ritchie.top>

点击进入[Ritchie's Blog](https://ritchie.top)


有悬浮的点击这里[Ritchie's Blog][any_flag]

[any_flag]: https://ritchie.top "Ritchie's Blog(optional)"


any_flag都懒得写的点击这里[Ritchie's Blog][]

[ritchie's blog]: https://ritchie.top "Ritchie's Blog(optional)"
```

让我们屏住呼吸，[一起回到最初的起点](#inner-link)。这个需要在页面指定的位置定义一个锚点，类似于这样`<a name='inner-line'></a>`

<https://ritchie.top>

<god@ritchie.top>

点击进入[Ritchie's Blog](https://ritchie.top)

有悬浮的点击这里[Ritchie's Blog][any_flag]

[any_flag]: https://ritchie.top "Ritchie's Blog(optional)"

any_flag都懒得写的点击这里[Ritchie's Blog][]

[ritchie's blog]: https://ritchie.top "Ritchie's Blog(optional)"

# 6. 图片

用法同插入超链接，区别在于最前面多个一个感叹号!

```text
markdown目前的插入图片语法，不能设置图片的具体参数，需要html标签配合支持

![图片缺失时显示的文字](https://ritchie.top/images/post/markdown.gif)

![福利不见了](https://ritchie.top/福利.jpeg)

[![带有链接的图片](https://ritchie.top/images/post/markdown2.gif)](http://ritchie.top)
```

![图片缺失时显示的文字](https://ritchie.top/images/post/markdown.gif)

![福利不见了](https://ritchie.top/福利.jpeg)

[![带有链接的图片](https://ritchie.top/images/post/markdown2.gif)](http://ritchie.top)

# 7. 列表

```text
- 无序列表
- 无序列表

    > 无序列表段落

    无序列表段落
- 无序列表
    - 无序列表
        - 无序列表
            - 无序列表

1. 有序列表
2. 有序列表

    有序列表段落

    有序列表段落
3. 有序列表

    > 1. 有序列表
    > 2. 有序列表

4. 有序列表
    - 无序列表
        - 无序列表
```

- 无序列表
- 无序列表

    > 无序列表段落

    无序列表段落
- 无序列表
    - 无序列表
        - 无序列表
            - 无序列表

1. 有序列表
2. 有序列表

    有序列表段落

    有序列表段落
3. 有序列表

    > 1. 有序列表
    > 2. 有序列表

4. 有序列表
    - 无序列表
        - 无序列表

# 8. 段落和换行

段落无首行缩进，段落之间通过**一个空行**分割（多个空行效果也是一样一样的）

一个段落内的换行，需要在一行的结尾添加**两个空格**（多写几个空格也是可以的）

# 9. 转义符

```text
**这是个粗体**

\*\*这不是个粗体\*\*
```

**这是个粗体**

\*\*这不是个粗体\*\*

# 10. 分割线

```text
睁大眼，瞧仔细，下面肯定有条分割线

* * * (最少三个)
```

* * *

# 11. 公式

```text
$$ y = \sum_{i=n}{x_i} $$

$$
\left[
\begin{matrix}
   1 & 2 & 3 \\\\
   4 & 5 & 6 \\\\
   7 & 8 & 9
\end{matrix}
\right] 
\tag{n}
$$

```

$$ y = \sum_{i=n}{x_i} $$

$$
\left[
\begin{matrix}
   1 & 2 & 3 \\\\
   4 & 5 & 6 \\\\
   7 & 8 & 9
\end{matrix}
\right] 
\tag{n}
$$

# 12. 表格

```text

| 默认对齐 | 居中对齐  | 右对齐 |
| --------|:-------:| -----:|
| 对齐的短线最少要写三个| 同列内对齐方式一致  | 烫烫烫 |
| 列宽是根据内容自适应的，这个可以很宽| **使用文本装饰**|不知道写啥了|

```

| 默认对齐 | 居中对齐  | 右对齐 |
| --------|:-------:| -----:|
| 对齐的短线最少要写三个| 同列内对齐方式一致  | 烫烫烫 |
| 列宽是根据内容自适应的，这个可以很宽| **使用文本装饰**|不知道写啥了|

# 13. 表情

```text

依赖插件：hexo-article-emoji
表情库：/node-modules/hexo-article-emoji/lib/data/full.json
用法：
    给大爷乐一个！:smile: :smiley: :grinning:

```

给大爷乐一个！:smile: :smiley: :grinning:

# 14. 脚注

```text

所有的脚注会自动放到整篇文章的末尾

子曰：靠！点脚注能跳转[^1]

[^1]: 《论语》

```

子曰：靠！点脚注能跳转[^1]

[^1]: 《论语》

# 15. 总结

前面对一些常用的语法进行简单的示例总结，后面再有新的需求再来补充。先这样！
