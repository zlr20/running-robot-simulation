# running-robot-simulation
更新日志：

- 2020.04.11：通过第一关（上下开横杆），主要策略是通过mobilenet网络进行二分类，识别横杆是否落下。等待横杆开启之后，以最快速度通过。
- 2020.04.11：用thu_Rst_Ruler_random控制器替换Rst_Ruler_random控制器，可以指定关卡。详询颜子琛。
- 2020.04.12：加了一个checkIfYaw函数，每一步根据陀螺仪信息判断是否总的偏航超过7.5°（默认，可调），如果是就通过转向修正回来（最好是调PID参数，目前是固定角速度转回来）
- 2020.04.12：通过第七关（水平开横杆），和第一关策略一样。
- 2020.04.12：将第一关、第七关所用的mobilenet网络训练代码放在`classification_mobilenet`目录中，解释文档可参考https://zlr20.github.io/2021/04/12/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0-1-%E8%87%AA%E5%AE%9A%E4%B9%89%E5%9B%BE%E5%83%8F%E5%88%86%E7%B1%BB/