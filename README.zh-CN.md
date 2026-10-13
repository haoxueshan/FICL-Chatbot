# 🤖 FICL Chatbot — RAG + LLM 聊天机器人学习示例

[한국어](README.md) | **简体中文**

本项目是一个 RAG（检索增强生成）聊天机器人示例：既可以在本地电脑运行 GGUF 量化的韩语大语言模型，也可以使用服务器上配置的 API 模型，并参考用户上传的文档生成回答。

> **学习目标**
> 1. 如何在普通电脑上运行 GGUF 量化模型
> 2. RAG（Retrieval-Augmented Generation）是什么，以及如何实现
> 3. 如何通过 SSE 实时流式输出大语言模型的回答

---

## 📦 项目结构

```
FICL_Chatbot/
├── chatbot.py              # FastAPI 后端：模型加载、RAG 和聊天流
├── ui.html                 # 中文、韩语网页界面，无需外部 CDN
├── guide.html              # 多语言使用指南，访问路径为 /guide
├── requirements.txt        # Python 依赖列表
├── requirements-api.txt    # 仅使用 API 模式时的依赖列表
├── api_models.example.json # 多个 API 服务的模型配置示例
├── api_models.json         # 服务器上的模型和密钥配置，不纳入 Git
├── start.bat               # Windows 启动脚本：自动安装并运行
├── start.sh                # Linux/Mac 启动脚本
├── models/                 # 自行下载的 GGUF 模型文件
├── rag_index/              # 自动生成的文档向量索引
├── uploads/                # 自动保存的上传原文件，不纳入 Git
├── README.md               # 韩语说明
└── README.zh-CN.md         # 本文件
```

---

## 🚀 Windows 快速开始

### ① 确认已安装 Python

- **要求版本**：Python 3.10 或更高版本
- 尚未安装时，从 [python.org/downloads](https://www.python.org/downloads/) 下载
- 安装时勾选 **“Add Python to PATH”**

### ② 下载模型（三选一）

把 GGUF 文件放入 `models/` 目录，文件名必须与下面列出的名称完全一致。

#### A) Qwen3-8B（多语言均衡型，推荐）

- **下载地址**：https://huggingface.co/bartowski/Qwen3-8B-GGUF
- **文件名**：`Qwen3-8B-Q4_K_M.gguf`
- **大小**：约 5.0 GB

#### B) EEVE-Korean-Instruct-10.8B（偏重韩语）

- **下载地址**：https://huggingface.co/heegyu/EEVE-Korean-Instruct-10.8B-v1.0-GGUF
- **下载文件**：`ggml-model-Q4_K_M.gguf`
- 下载后重命名为 **`EEVE-Korean-Instruct-10.8B-v1.0-Q4_K_M.gguf`**
- **大小**：约 6.8 GB

#### C) EXAONE-3.5-7.8B（LG AI，偏重韩语）

- **下载地址**：https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF
- **文件名**：在页面中选择 `EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf`
- **大小**：约 4.9 GB

> 💡 初次使用可以先选体积较小的 EXAONE 7.8B。电脑配置较高时，也可以尝试 EEVE 10.8B。

> ⚠ 下载某些模型可能需要免费注册 Hugging Face 账号。

### ③ 启动

双击 `start.bat`。首次运行时，脚本会：

1. 创建 `.venv` 虚拟环境
2. 安装 `requirements.txt` 中的依赖，约需 5～10 分钟
3. 检查模型文件
4. 在 http://localhost:7861 启动服务器

### ④ 使用

在浏览器中打开 **http://localhost:7861**：

- 从顶部下拉菜单选择已下载的模型
- 按需上传 PDF、DOCX 等文件以建立索引
- 输入消息并按 Enter

---

## 🛠 手动安装

如果不使用 `start.bat`，可以按以下步骤创建虚拟环境并安装依赖。

### Windows

```bat
REM 1) 创建虚拟环境
python -m venv .venv

REM 2) 激活虚拟环境
.venv\Scripts\activate

REM 3) 升级 pip
python -m pip install --upgrade pip

REM 4) 安装依赖
pip install -r requirements.txt

REM 5) 启动
python chatbot.py
```

### Linux / Mac

```bash
# 1) 创建虚拟环境
python3 -m venv .venv

# 2) 激活虚拟环境
source .venv/bin/activate

# 3) 升级 pip
pip install --upgrade pip

# 4) 安装依赖
pip install -r requirements.txt

# 5) 添加执行权限并启动
chmod +x start.sh
./start.sh
# 或者
python chatbot.py
```

---

## ⚡ GPU 加速（可选）

仅使用 CPU 也能运行。如果电脑有 NVIDIA GPU，可以通过 GPU 加速模型推理。

### Windows + NVIDIA GPU

默认安装的 `llama-cpp-python` 使用 CPU。如需 CUDA 支持，请替换为相应版本：

```bat
REM 激活虚拟环境
.venv\Scripts\activate

REM 卸载 CPU 版本
pip uninstall llama-cpp-python -y

REM 安装 CUDA 12.1 预编译包
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

使用 CUDA 11.8 时，将 URL 末尾的 `cu121` 改为 `cu118`。

> 在命令提示符中运行 `nvidia-smi`，可以查看显示的 CUDA Version。

### 调整 GPU 层数

默认将所有层加载到 GPU（`N_GPU_LAYERS=-1`）。如果显存不足，可以只加载一部分：

```bat
REM CMD
set N_GPU_LAYERS=20
python chatbot.py

REM PowerShell
$env:N_GPU_LAYERS=20
python chatbot.py
```

| 显存 | 建议的 `N_GPU_LAYERS` |
|---|---|
| 24 GB（RTX 3090/4090） | -1（全部） |
| 12 GB（RTX 3060/4070） | 25～30 |
| 8 GB | 15～20 |
| 4 GB 或没有 GPU | 0（仅 CPU） |

---

## 💡 使用方法

### 1. 选择语言和模型

在页面顶部选择 **中文 / 한국어**，界面文字及后续回答的语言会随之改变。语言选择保存在浏览器中，已有对话不会自动翻译。然后在模型下拉菜单中选择服务器提供的模型；本地 GGUF 模型和服务器配置的 API 模型会显示在同一个列表中。

- 本地 GGUF：只有已下载的模型可以选择。
- API 模型：只有已在服务器登记且设置了 API Key 的模型可以选择。

浏览器只向服务器发送模型 ID。API Key、Base URL 和实际 API 模型名称仅由服务器配置管理。

#### 在服务器上配置 API 模型

将 [api_models.example.json](api_models.example.json) 复制为 `api_models.json`，再设置各服务的 `base_url`、`models` 和 `api_key`。服务器直接从文件读取 `api_key`，不会从环境变量读取密钥。一个服务下可以配置多个模型。

```json
[{"id":"example","name":"Example API","base_url":"https://api.example.com/v1","api_key":"YOUR_KEY","models":["example-small","example-large"]}]
```

`api_models.json` 已被 Git 忽略，浏览器 API 也不会返回密钥。请只允许服务器管理员读取该文件，不要分享或提交密钥。仅为支持该功能的 API 模型（例如 Qwen）设置 `enable_thinking`。修改配置后，重新检查 `/status` 中的模型列表。API 模型的 `available` 仅表示已设置密钥，不代表实际请求已经通过认证。

本地模型和 API 模型共用一个选择列表。`/status` 中的 `default_model` 会优先使用可用的 `DEFAULT_MODEL`；如果该模型不可用，则选择第一个可用模型。明确选择某个本地模型而其文件不存在时，服务会返回错误，不会自动换用其他模型。选择 API 模型后，可以点击 **测试 API 连接**，验证认证信息和模型 ID。该按钮会发送一条简短的真实请求，可能产生少量 API 费用，不会自动执行。

如果没有 GGUF 文件，但已配置 API 模型，`start.bat` / `start.sh` 会安装不含本地推理库的 `requirements-api.txt`。手动安装时也可以运行 `pip install -r requirements-api.txt`。

外部 API 需使用 HTTPS；只有 localhost 允许使用 HTTP。选择 API 模型后，问题及检索到的 RAG 文档内容会发送给对应的 API 服务。若将 `HOST` 设为 `0.0.0.0` 等地址以供局域网访问，必须设置长度至少为 16 个字符的 `APP_ACCESS_TOKEN`，否则服务器不会启动。首次访问时，浏览器会要求输入令牌，并仅在当前浏览器会话中保存。没有 HTTPS 的局域网会以明文 HTTP 传输令牌，因此只应在可信网络中使用，或部署 HTTPS 反向代理。

### 2. 上传文件（可选）

点击 **📁 上传文件**，选择 PDF、DOCX、XLSX、TXT 或 MD 文件。

- 系统自动提取文本、切分文本块、生成嵌入向量并保存索引。
- 不同文件名的文件会追加到现有索引；重新上传同名文件只会替换该文件的旧内容。
- 单个文件最大 20 MB；一次最多 10 个文件，合计不超过 50 MB。
- 原文件保存在 `uploads/` 中，可以从网页的上传文件列表逐个删除；删除时，对应的检索文本块也会移除。
- 索引持久保存在 `rag_index/index.npz`，重启后仍可使用；旧格式 `docs.json` + `embeddings.npy` 也可读取。
- 对于旧索引中有记录、但没有保存原文件的资料，列表会显示 **原文件未保存**，仍可删除对应的检索文本块。

### 3. 提问

在输入框中输入问题并按 Enter；Shift+Enter 用于换行。系统会自动检索已上传资料并参考它们回答，在回答下方显示 **📚 参考资料：文件名**。

### 4. 删除文件或清空所有资料

在网页文件列表中点击 **删除**，会同时删除该原文件及其索引文本块。**清空所有资料** 会删除全部原文件和索引，请确认后使用。

---

## ⚙️ 环境变量（可选）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DEFAULT_MODEL` | `qwen3-8b` | 默认模型键：本地键或 `api:provider:model-id`；不可用时选择第一个可用模型 |
| `HOST` | `127.0.0.1` | 监听地址，局域网共享时可设为 `0.0.0.0` |
| `APP_ACCESS_TOKEN` | 空 | 局域网开放访问时必需的网页访问令牌，与服务器 API Key 不同 |
| `PORT` | `7861` | 端口号 |
| `N_GPU_LAYERS` | `-1` | 加载到 GPU 的层数；`-1` 表示全部，`0` 表示仅 CPU |
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | RAG 嵌入模型 |

设置示例：

```bat
REM Windows
set DEFAULT_MODEL=eeve-10.8b
set N_GPU_LAYERS=25
python chatbot.py
```

```bash
# Linux/Mac
DEFAULT_MODEL=exaone-7.8b N_GPU_LAYERS=0 python chatbot.py
```

---

## 🧠 代码结构

### 后端：[chatbot.py](chatbot.py)

| 部分 | 函数 | 作用 |
|---|---|---|
| 加载大语言模型 | `get_llm()` | 用 `Llama()` 加载 GGUF 文件；使用锁保护和快速路径 |
| 文本切块 | `chunk_text()` | 将长文档按 500 字符切块，相邻块重叠 80 字符 |
| 文本提取 | `extract_text()` | 从 PDF、DOCX、XLSX、TXT、MD 中提取正文 |
| 构建索引 | `build_index()` | 为文本块生成嵌入向量，保存单个 NPZ 索引和原文件 |
| 模型选择 | `get_model_catalog()` / `select_model()` | 提供状态查询和聊天共用的模型列表，并验证选择 |
| 检索 | `search_rag_per_doc()` | 按文档进行余弦相似度检索 |
| 提示词 | `build_messages()` | 组合系统消息、参考资料、对话历史和当前问题 |
| 流式输出 | `/chat` 接口 | 通过 SSE 逐个输出 token |

### 前端：[ui.html](ui.html)

- 单个 HTML/JavaScript 页面，不依赖框架或 CDN；安装后界面可离线运行
- 通过 `fetch()` + `ReadableStream` 直接解析 SSE
- 支持中文、韩语界面和回答语言选择、模型选择、文件上传与列表、逐个删除、清空资料以及聊天

### RAG 工作流程

```
[上传文件]
    ↓
1. extract_text()：文档 → 纯文本
    ↓
2. chunk_text()：纯文本 → 500 字符的文本块
    ↓
3. embedder.encode()：文本块 → 768 维嵌入向量
    ↓
4. 保存到磁盘，供重启后使用

[聊天]
    ↓
1. 用户消息 → embedder.encode() → 问题向量
    ↓
2. 已保存的嵌入向量与问题向量计算余弦相似度
    ↓
3. 将相关度最高的文本块加入系统提示词
    ↓
4. 调用大语言模型 → 逐个 token 流式输出回答
```

---

## ❓ 常见问题

### 提示“未安装 Python”

从 [python.org](https://www.python.org/downloads/) 安装，并勾选将 Python 加入 PATH。

### Windows 上安装 `llama-cpp-python` 失败

可能需要编译工具，可以尝试以下方法：

```bat
REM 方法 A：优先使用预编译包
pip install llama-cpp-python --prefer-binary

REM 方法 B：安装 Visual Studio Build Tools 2022 后重试
REM https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

### 提示“找不到模型文件”

- 确认 GGUF 文件已放入 `models/`
- 确认文件名完全一致，包括大小写
- 如果下载中断，重新下载文件

### 首次运行 `sentence-transformers` 很慢

- 程序会自动下载约 500 MB 的嵌入模型，需要联网
- 下载完成后，后续启动会更快

### 回答速度太慢

1. 启用 GPU 加速，参见上文“GPU 加速”
2. 使用较小的模型，例如 EXAONE 7.8B
3. 仅使用 CPU 时，可以在 `chatbot.py` 中按 CPU 核心数调整 `n_threads`

### 端口 7861 已被占用

通过环境变量更换端口：`set PORT=8000`。

---

## 📚 外部模型许可证

| 模型 | 许可证 | 商业使用 |
|---|---|---|
| Qwen3 | Apache 2.0 | 可使用 |
| EEVE-Korean | Apache 2.0；底座 SOLAR 有单独许可证 | 需核对底座许可证 |
| EXAONE | LG AI Research 许可证 | 商业使用前需核对 |

使用外部模型前，请阅读相应模型页面上的最新许可证条款。

---

## 🎯 后续学习方向

理解这个示例后，可以尝试：

1. **更换嵌入模型**：将 `intfloat/multilingual-e5-base` 换成 `BAAI/bge-m3` 等
2. **增加重排序**：使用 cross-encoder 对初步检索结果重新排序
3. **混合检索**：结合 BM25 关键词检索与嵌入向量语义检索
4. **对话记忆摘要**：让大语言模型压缩长对话，保留上下文
5. **函数调用**：接入天气 API 等外部工具

---

反馈问题时，请附上控制台日志及 `/status` 的响应。
