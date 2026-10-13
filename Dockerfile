# 一个能一键起服务的镜像。
#
# 为什么不把依赖装到宿主机就行：本项目要 Python 3.13 + 一批固定版本的包。
# 别人机器上可能是 3.11，或者 pip 装的时候版本冲突。镜像把这些差异抹平了。
FROM python:3.13-slim

WORKDIR /app

# 先拷依赖清单再装：这样改代码不会让依赖层缓存失效，重建快得多
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 让 print 立刻输出，否则 docker logs 里看不到进度
ENV PYTHONUNBUFFERED=1

# 每轮工具/模型调用的说明写在这里，方便别人知道这个镜像怎么用
LABEL org.opencontainers.image.title="tax-agent-harness" \
      org.opencontainers.image.description="企业税负优化多智能体 Agent 运行时" \
      org.opencontainers.image.source="https://github.com/tk-yuan/tax-agent-harness"

EXPOSE 8000

# 默认起服务；想跑 CLI 就 docker run --rm -it <image> python scripts/run.py "目标"
CMD ["python", "scripts/serve.py", "0.0.0.0", "8000"]
