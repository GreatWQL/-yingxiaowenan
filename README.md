# 英语课程营销文案工坊

一键生成朋友圈/海报/短视频招生文案。基于 Streamlit + OpenAI 兼容接口，支持 Kimi（Moonshot）与 DeepSeek。

## 本地运行
1. Python 3.10+
2. 配置环境变量（任选其一或同时配置）：
   - Kimi（Moonshot）：`export MOONSHOT_API_KEY=sk-xxxx`
   - DeepSeek：`export DEEPSEEK_API_KEY=sk-xxxx`
3. 安装依赖并运行：
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## 提供商选择
- 在侧边栏选择 `LLM 提供商`：`Kimi（Moonshot）` 或 `DeepSeek`。
- 默认模型：
  - Kimi：`moonshot-v1-8k`（国内 `https://api.moonshot.cn/v1`）
  - DeepSeek：`deepseek-chat`（`https://api.deepseek.com/v1`）

## 合规
- 输出严禁绝对化承诺；若检测到敏感词，文案中会高亮标注并在页面提示。
- 页面底部提供统一合规提醒，请在发布前人工审核。

## 部署
- Hugging Face Spaces 或 Streamlit Community Cloud（推荐用于运行 Streamlit）。
- 在部署平台的 Secrets 中配置：`MOONSHOT_API_KEY` 或 `DEEPSEEK_API_KEY`。
- Cloud Secrets 读取规则：若未设置环境变量，应用会自动从 `st.secrets` 读取同名键（`MOONSHOT_API_KEY`/`DEEPSEEK_API_KEY`），亦支持 `st.secrets['api_keys'][ENV]` 或 `st.secrets['moonshot']['api_key']`、`st.secrets['deepseek']['api_key']`。

### 在 Vercel 展示（两种方式）
1) 反向代理（推荐）
   - 在仓库根目录添加 `vercel.json`：
     ```json
     {
       "rewrites": [
         { "source": "/", "destination": "https://YOUR_STREAMLIT_HOST/" },
         { "source": "/(.*)", "destination": "https://YOUR_STREAMLIT_HOST/$1" }
       ]
     }
     ```
   - 将 `YOUR_STREAMLIT_HOST` 替换为你在 Spaces/Streamlit Cloud 的公开链接（不带路径）。
   - Vercel 不需要配置 API Key，密钥仍保留在实际运行平台（Spaces/Streamlit Cloud）。
   - 若选择此方式，请不要保留 `public/index.html`，否则根路径会优先返回静态文件而不是代理到外部主机。

2) iframe 嵌入
   - 使用 `public/index.html`，其中 `<iframe src="https://YOUR_STREAMLIT_HOST/">` 指向你的公开链接。
   - 部署到 Vercel 后即可在你的域名内嵌外部应用。

## 常见问题
- 看不到“生成”：检查所选提供商对应的 API Key 是否已在运行平台配置（Vercel 仅代理或嵌入，无需密钥）。
- 生成失败：在运行平台检查日志；或切换提供商、降低温度稍后重试。