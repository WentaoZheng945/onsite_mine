# python环境
FROM python:3.9

# 复制基础内容
COPY . .

# 查看python版本
RUN python3 --version

# 安装Tkinter依赖
# RUN apt-get update && apt-get install -y python3-tk

# 更新pip
RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装测试环境
RUN pip install onsite-mine==0.1.5 -i https://pypi.tuna.tsinghua.edu.cn/simple