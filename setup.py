#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： Wentao Zheng
# datetime： 2023/11/17 14:47 
# ide： PyCharm

from setuptools import setup, find_packages

with open('./setup/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='onsite_mine',  # 包名
    version='0.0.1',  # 版本
    description="The simulation environment for unstructured road replay testing in onsite competitions.",  # 包简介
    long_description=open('README.md').read(),  # 读取文件中介绍包的详细内容
    include_package_data=True,  # 是否允许上传资源文件
    author='Zheng Wentao',  # 作者
    author_email='swjtu_zwt@163.com',  # 作者邮件
    maintainer='ZhengWentao',  # 维护者
    maintainer_email='swjtu_zwt@163.com',  # 维护者邮件
    license='MIT License',  # 协议
    url='https://github.com/WentaoZheng945/onsite_mine',  # github或者自己的网站地址
    packages=find_packages(),  # 用setuptools找到项目所有有关的包列表
    classifiers=[
        'Development Status :: 3 - Alpha',  # 包的开发状态，
        # Alpha 表示软件处于开发的早期阶段，可能仍然不够稳定，可能会有较大的变化。其他可能的值包括 Beta（测试阶段）、Production/Stable（稳定版）等。
        'Intended Audience :: Developers',  # 预期用户群
        # 这里指定了该包的预期用户为开发者。其他可能的值包括 End Users（终端用户）、Science/Research（科研人员）等。
        'Topic :: Software Development :: Testing',  # 主题和领域。
        # 解释：这里指定了该包与软件开发和测试相关。其他可能的值包括 Web（Web 开发）、Database（数据库相关）等
        'License :: OSI Approved :: MIT License',  # 许可协议
        'Programming Language :: Python :: 3',  # 设置编写时的python版本
    ],
    python_requires='>=3.7',  # 设置python版本要求
    install_requires=requirements,  # 安装所需要的库
    # entry_points={
    #     'console_scripts': [
    #         ''],
    # },  # 设置命令行工具(可不使用就可以注释掉)

)