# gym-chinese-chess
一个简单的中国象棋gym环境，可以产出类似Alpha zero算法需要的带有历史盘面的observation.

# 安装
运行如下指令安装中国象棋gym环境

```shell script
pip install -e .
```

# 中国象棋gym 101
阅读 [gym_101.ipynb](ipynbs/gym_101.ipynb) 查看一些简单的使用例子.

在安装完gym-chinese-chess之后你可以简单的使用下面命令创建一个中国象棋环境
```python
import gym
chinese_chess_env = gym.make('gym_chinese_chess:cchess-v0')
```

并且可以通过
```python
print(chinese_chess_env.render())
```
来展示当前盘面：

```text
 9俥傌象士将士象傌俥
 8．．．．．．．．．
 7．砲．．．．．砲．
 6卒．卒．卒．卒．卒
 5．．．．．．．．．
 4．．．．．．．．．
 3兵．兵．兵．兵．兵
 2．炮．．．．．炮．
 1．．．．．．．．．
 0车马相仕帅仕相马车
  ａｂｃｄｅｆｇｈｉ
```

可以通过

```python
actions = chinese_chess_env.get_possible_actions()
```

获取action space中的所有当前局面下的可能action的integer list

也可以通过
```python
moves = chinese_chess_env.get_possible_moves()
```
来取得局面下所有可能action的字符串形式

然后通过
```python
state, reward, done, info = chinese_chess_enf.step(action)
```
的方式获得gym规范的state，reward，游戏是否结束标志done和一些调试信息info。

如果done为True，则游戏已经结束，

其他中国象棋gym的用法在[gym_101.ipynb](ipynbs/gym_101.ipynb)中有列举,请参考其中列举的方法，相信你可以很快上手。

# TODOs
1. 本项目实现的规则与中国象棋亚洲规则/中国规则有一些区别，主要体现在长将/长捉的判断上，本项目出于方便考虑，并没有准确实现长将/长捉逻辑，本项目使用同局三现时若最后一个移动的子力非将军则判负这一简单逻辑来替代长将/长捉判断,这个逻辑在一些情况下仍然是很有问题的，之后如果有需求或者有时间会考虑更改替,也希望大家如果有兴趣可以提pr解决这个问题。