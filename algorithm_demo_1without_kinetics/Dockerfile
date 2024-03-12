FROM zhengwentao/onsite-mine-image-base:0.0.2

# 复制算法内容
COPY ./planner ./planner

# 运行运行脚本
ENTRYPOINT python /planner

# 尽量不使用CMD，防止作弊，同时统一格式
# CMD ["python3", "planner/__main__.py"]