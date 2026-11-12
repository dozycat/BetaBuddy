# BetaBuddy Web（不再维护）

BetaBuddy 最早的形态：一个跑在电脑上的网页应用。上传攀岩视频，由 Python 服务用 MediaPipe 做姿态识别、计算重心与关节角度、生成带骨架的视频，再通过本机的 Ollama 模型给出文字建议。

它和 iPhone 应用没有代码或运行时上的关系，现在不再维护，也不保证还能跑起来。放在这里是为了保留历史和可读的参考实现。iPhone 应用见[仓库首页](../../README.md)。

## 里面有什么

- `backend/`：FastAPI 服务。视频上传、姿态识别、指标计算、动作检测、视频标注、Ollama 调用，SQLite 存储。
- `frontend/`：React + TypeScript + Vite 的单页应用。
- `docker-compose.yml`、`nginx.conf`：容器部署。
- `startup.sh`、`shutdown.sh`：在本机装依赖并启动、停止两个服务。
- `climbing-analysis-design.md`：当时的设计文档。
- `docs/introduction.mp4`：当时的介绍视频。

## 想跑一下

需要 Python 3.10+、Node.js 18+；文字建议需要本机的 [Ollama](https://ollama.ai)，没有时退回规则建议。所有命令在本目录下执行。

```sh
./startup.sh      # 后端 http://localhost:8000，前端 http://localhost:5173
./shutdown.sh
```

或者分别启动：

```sh
cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

cd frontend && npm install && npm run dev
```

这套实现里的「稳定性评分」「效率」「动作识别」等指标没有经过验证。iPhone 应用有意不沿用它们，原因见 [分析诚实性](../../docs/ios/product-design.md)。
