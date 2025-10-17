import os
import re
from pathlib import Path
import streamlit as st
from openai import OpenAI

# ------------------------
# 配置
# ------------------------
APP_TITLE = "英语课程营销文案工坊"
APP_DESC = "一键生成朋友圈/海报/短视频等多渠道招生文案，附CTA与合规提示。"

# 双提供商映射（Moonshot Kimi / DeepSeek）
PROVIDERS = {
    "Kimi（Moonshot）": {
        "base_url": "https://api.moonshot.cn/v1",
        "model": "moonshot-v1-8k",
        "env": "MOONSHOT_API_KEY",
        "desc": "适合中文场景，8k上下文"
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "env": "DEEPSEEK_API_KEY",
        "desc": "通用对话模型，性价比较高"
    },
}

# ------------------------
# 工具函数
# ------------------------
@st.cache_data(show_spinner=False)
def load_system_prompt() -> str:
    p = Path("prompts/system_prompt.txt")
    if p.exists():
        return p.read_text(encoding="utf-8")
    # 兜底简版
    return (
        "你是教育行业资深市场文案。请根据输入生成不同渠道的中文招生文案，包含统一CTA与合规提醒。"
    )


def get_client(provider_name: str) -> OpenAI:
    cfg = PROVIDERS.get(provider_name)
    if not cfg:
        st.sidebar.error("未知提供商，请在侧边栏重新选择。")
        raise ValueError("Invalid provider")
    env_key = cfg["env"]
    # 读取顺序：环境变量 -> st.secrets 同名 -> st.secrets['api_keys'][ENV] -> st.secrets['moonshot'/'deepseek']['api_key']
    api_key = os.environ.get(env_key)
    if not api_key:
        try:
            api_key = st.secrets.get(env_key) or st.secrets.get("api_keys", {}).get(env_key)
            if not api_key:
                provider_key = "moonshot" if "MOONSHOT" in env_key else ("deepseek" if "DEEPSEEK" in env_key else None)
                if provider_key:
                    api_key = st.secrets.get(provider_key, {}).get("api_key")
        except Exception:
            api_key = None
    if not api_key:
        st.sidebar.warning(f"未检测到 {env_key}（环境变量或Secrets）。请到部署平台配置。")
    return OpenAI(base_url=cfg["base_url"], api_key=api_key)


def build_user_prompt(data: dict) -> str:
    return f"""
课程名：{data['course_name']}
年龄/级别：{data['age_level']}
班型与课时：{data['format_hours']}
时间与地点：{data['time_place']}
核心卖点（分条）：{data['selling_points']}
家长痛点（分条）：{data['parent_pains']}
师资人设：{data['teacher_persona']}
品牌风格：{data['brand_style']}
语气强度：{data['tone_strength']}
是否加入Emoji：{data['use_emoji']}
禁用词：{data['banned_words']}
渠道：{", ".join(data['channels'])}
每类生成条数：{data['n_per_channel']}
请严格按渠道分别输出内容，并在最后附统一CTA模板与合规提醒。
"""


def detect_banned_words(text: str, banned_words: list[str]) -> list[str]:
    hit = []
    for w in banned_words:
        w = w.strip()
        if not w:
            continue
        if re.search(re.escape(w), text, flags=re.IGNORECASE):
            hit.append(w)
    return hit


def sanitize_banned_words(text: str, banned_words: list[str]) -> str:
    # 轻度净化：高亮提示但不硬替换，避免误伤语义
    for w in banned_words:
        if not w.strip():
            continue
        text = re.sub(re.escape(w), f"【合规敏感词：{w}】", text, flags=re.IGNORECASE)
    return text

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption(APP_DESC)

with st.sidebar:
    provider_name = st.selectbox("LLM 提供商", list(PROVIDERS.keys()), index=0)
    st.caption(f"当前模型：{PROVIDERS[provider_name]['model']} — {PROVIDERS[provider_name]['desc']}")

    st.subheader("课程信息")
    course_name   = st.text_input("课程名", "S2自然拼读进阶班")
    age_level     = st.text_input("年龄/级别", "7-9岁 / S2")
    format_hours  = st.text_input("班型与课时", "小班 12次×90min")
    time_place    = st.text_input("时间与地点", "每周六 10:00-11:30 / XX校区")
    selling_points= st.text_area("核心卖点（换行分条）", "系统Phonics\n大量跟读纠音\n分层作业\n家校共育")
    parent_pains  = st.text_area("家长痛点（换行分条）", "发音不准\n读词靠猜\n记词低效")

    st.subheader("风格与合规")
    teacher_persona = st.selectbox("师资人设", ["亲和", "专业", "学术", "幽默"])
    brand_style     = st.selectbox("品牌风格", ["热烈", "理性", "温情", "权威"])
    tone_strength   = st.selectbox("语气强度", ["稳健", "适中", "强势"])
    use_emoji       = st.selectbox("是否加入Emoji", ["是", "否"])
    banned_words    = st.text_input("禁用词（逗号分隔）", "保分, 速成, 100%, 包通过")

    st.subheader("生成设置")
    channels = st.multiselect(
        "选择渠道", ["朋友圈", "海报", "短视频", "小红书"],
        default=["朋友圈", "海报", "短视频"]
    )
    n_per_channel = st.slider("每类生成条数", 1, 5, 3)

    col_a, col_b = st.columns(2)
    with col_a:
        gen = st.button("生成文案", type="primary")
    with col_b:
        if st.button("填充示例"):
            st.session_state.update({
                "course_name": "S1自然拼读启蒙班",
                "age_level": "6-8岁 / S1",
                "format_hours": "走读 10次×90min",
                "time_place": "周日 10:00-11:30 / XX校区",
            })

# 将sidebar变量打包
data = dict(
    course_name=course_name,
    age_level=age_level,
    format_hours=format_hours,
    time_place=time_place,
    selling_points=selling_points,
    parent_pains=parent_pains,
    teacher_persona=teacher_persona,
    brand_style=brand_style,
    tone_strength=tone_strength,
    use_emoji=use_emoji,
    banned_words=banned_words,
    channels=channels,
    n_per_channel=n_per_channel,
)

# ------------------------
# 主逻辑
# ------------------------
if gen:
    if not channels:
        st.warning("请至少选择一个渠道。")
        st.stop()

    sys_prompt = load_system_prompt()
    user_prompt = build_user_prompt(data)
    cfg = PROVIDERS[provider_name]
    client = get_client(provider_name)

    with st.spinner("正在生成文案..."):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                top_p=0.95,
            )
            out = resp.choices[0].message.content
        except Exception as e:
            st.error(f"生成失败：{e}")
            st.stop()

    # 合规提示（检测禁用词）
    banned_list = [w.strip() for w in banned_words.split(",")]
    hits = detect_banned_words(out, banned_list)
    if hits:
        st.warning("检测到潜在敏感词（已在文案中标注）： " + "、".join(hits))
        out = sanitize_banned_words(out, banned_list)

    st.subheader("生成结果")
    st.markdown(out)

    st.download_button(
        "下载为TXT",
        data=out,
        file_name="marketing_copy.txt",
        mime="text/plain",
    )

st.markdown("---")
st.caption("合规提醒：学习效果因人而异，需持续投入；本工具输出仅作参考文案，最终发布前请再次人工审核。")