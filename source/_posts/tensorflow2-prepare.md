---
title: 【Tensorflow2】准备工作
mathjax: true
toc: false
comments: true
date: 2020-01-11 11:55:48
categories: tensorflow
tags: pyenv pycharm
---

> 兵马未动，粮草先行 —— 孟子

其实也没啥好准备的，把Tensorflow装上就行了。之所以单独搞一篇，除了刷量之外，还希望给大家推荐一些比较好用的工具，也方面后面开发过程中使用。

<!--more-->

## 什么是Tensorflow

客官这边请～[百度百科](https://baike.baidu.com/item/Tensorflow/18828108) [官网](https://tensorflow.google.cn/)

## 为啥是Tensorflow

最流行啊、最全面啊、最强大啊

PyTorch？好走不送。[PyTorch官网](https://tensorflow.google.cn/)

## 安装配置

冥冥之中，你还是选择了Tensorflow，Tensorflow也正是为你而生！

安装不能再简单
```
pip install tensorflow==2.0.0
```
Hold On!
好像现实中，大家并没有这么愉快的结束这一切。所以，跟着我的节奏，咱们重新开始安装。

### 安装pyenv

> 一个能让你有勇气pip install任何工具包的版本管理工具

pyenv是python版本的管理工具，你可以借助pyenv创建各种python虚拟环境。在执行python代码之前，通过激活指定的python虚拟环境，来实现python代码执行环境的切换。这样妈妈就再也不用担心我把系统python的环境搞咋了。
pyenv的安装分为Linux版和Mac版（Windows？咱当作娱乐工具就行了）

Linux版安装步骤：
1. sudo apt install git
2. git clone https://github.com/yyuu/pyenv.git ~/.pyenv
3. 修改`~/.bashrc`或者`~/.profile`文件，添加如下内容
    ```shell
    export PATH=~/.pyenv/bin:$PATH
    export PYENV_ROOT=~/.pyenv
    eval "$(pyenv init -)"
    ```
4. source `~/.bashrc`（或者`~/.profile`）

Mac版安装步骤：
1. brew install pyenv
2. 修改`~/.zshrc`，添加如下内容
    ```shell
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
    ```
3. source `~/.zshrc`

### pyenv创建虚拟环境

1. 安装python
    pyenv install 3.5.9
2. 创建虚拟环境
    pyenv virtualenv 3.5.9 tensorflow2_py3
3. 激活虚拟环境
    pyenv activate tensorflow2_py3

pyenv本身的命令还是挺多的，不过上面三个命令应该是最常用的了。更多命令，可以自行搜索pyenv教程。

有了pyenv的加持，tensorflow的安装就变得很简单了。如果后面在安装中碰到依赖冲突、依赖缺失等问题，就可以随心所欲的切换python环境了。

### Jupyter安装和使用

jupyter是一个超级好用的python开发工具，能够让我们在网页上写代码，带来快捷高效的开发测试体验。

> pip install jupyter

如果这条命令没有想象中那么顺利，那就循着错误信息一步一步解决吧。因为我们拥有了pyenv，所以任何解决错误的尝试，尽管去试吧！

> jupyter notebook

这个命令将启动jupyter服务，并自动打开工作台的网址。具体怎么使用，大家自己慢慢探索尝试吧，一定不会让你失望的！

## 选择一款好的IDE

jupyter固然很好，但她毕竟只是一个调试工具。真正开发项目，还是需要一款优秀的IDE作为后勤支持。
这里我推荐两款IDE：PyCharm（开箱即用，体积稍大）和VS Code（需要配置，体积轻便）。
细节不多说，谁用谁知道。

---

准备工作到这里应该就差不多了，我们拥有了Python开发环境、Tensorflow也已经安装就位、方便的测试工具Jupyter配上一款优秀的IDE。
目前暂时想不到还有什么需要补充的，可能还有一些小工具。那就用到的时候再一一介绍吧。
所谓工欲善其事，必先利其器。我们利器已经打磨完毕，Tensorflow的征程正式起航！
