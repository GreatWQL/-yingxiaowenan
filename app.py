import os
import re
from pathlib import Path
import streamlit as st
from openai import OpenAI
import io
import urllib.request
from PIL import Image, ImageDraw, ImageFont
import replicate

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
# 生成扩展：Replicate/占位模式
# ------------------------

def ensure_replicate_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        try:
            token = st.secrets.get("REPLICATE_API_TOKEN") or st.secrets.get("replicate", {}).get("api_token")
        except Exception:
            token = None
    if token:
        os.environ["REPLICATE_API_TOKEN"] = token
    else:
        st.sidebar.info("未检测到 REPLICATE_API_TOKEN（环境变量或Secrets）。可使用演示模式生成占位图片/视频。")
    return token


def run_replicate_model(model_slug: str, input_payload: dict):
    token = ensure_replicate_token()
    if not token:
        raise RuntimeError("缺少 REPLICATE_API_TOKEN")
    client = replicate.Client(api_token=token)
    model = client.models.get(model_slug)
    versions = list(model.versions.list())
    version = versions[0] if versions else None
    if version is None:
        raise RuntimeError("无法获取模型版本")
    return client.run(version, input=input_payload)


def make_placeholder_image(text: str, width: int = 1024, height: int = 1024, bg=(245, 245, 245)):
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    # 简单换行布局
    words = text.split()
    lines, line = [], ""
    for w in words:
        test = (line + " " + w).strip()
        if draw.textlength(test, font=font) < width - 40:
            line = test
        else:
            if line:
                lines.append(line)
            line = w
    if line:
        lines.append(line)
    y = 50
    for ln in lines[:25]:
        draw.text((20, y), ln, fill=(20, 20, 20), font=font)
        y += 20
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def make_placeholder_gif(text: str, frames: int = 24, width: int = 640, height: int = 360):
    images = []
    font = ImageFont.load_default()
    for i in range(frames):
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(10, 10), (width - 10, height - 10)], outline=(0, 0, 0), width=2)
        draw.text((20 + i * 5, height // 2 - 10), text[:200], fill=(0, 0, 0), font=font)
        images.append(img)
    buf = io.BytesIO()
    images[0].save(buf, format="GIF", save_all=True, append_images=images[1:], duration=80, loop=0)
    buf.seek(0)
    return buf

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

# ------------------------
# 图片生成（Replicate + 演示模式）
# ------------------------
st.subheader("图片生成（Beta）")
col_img_a, col_img_b, col_img_c = st.columns([2, 1, 1])
with col_img_a:
    img_prompt = st.text_input("图片提示词（中文/英文均可）", "儿童英语拼读课堂，温暖、亲和、明亮")
with col_img_b:
    img_w = st.selectbox("宽度", [512, 768, 1024], index=2)
with col_img_c:
    img_h = st.selectbox("高度", [512, 768, 1024], index=2)
col_img_d, col_img_e = st.columns([1, 1])
with col_img_d:
    img_n = st.slider("数量", 1, 4, 2)
with col_img_e:
    img_demo = st.checkbox("演示模式（无API）", value=False, help="无密钥时生成占位图")
img_model_slug = st.text_input("模型（Replicate slug）", "stability-ai/stable-diffusion-xl")
btn_img = st.button("生成图片", type="primary")

if btn_img:
    try:
        if img_demo:
            st.info("演示模式：生成占位图片")
            for i in range(img_n):
                buf = make_placeholder_image(f"{img_prompt} #{i+1}", width=img_w, height=img_h)
                st.image(buf, caption=f"占位图 {i+1}")
                st.download_button(f"下载占位图 {i+1}", data=buf.getvalue(), file_name=f"placeholder_{i+1}.png", mime="image/png")
        else:
            outputs = run_replicate_model(img_model_slug, {
                "prompt": img_prompt,
                "num_outputs": img_n,
                "width": img_w,
                "height": img_h,
            })
            # outputs 可能是生成器或列表
            urls = list(outputs) if not isinstance(outputs, list) else outputs
            if not urls:
                st.warning("未返回图片URL，请更换模型或稍后重试。")
            for i, url in enumerate(urls, start=1):
                st.image(url, caption=f"生成图 {i}")
                try:
                    data = urllib.request.urlopen(url).read()
                    st.download_button(f"下载图片 {i}", data=data, file_name=f"image_{i}.png", mime="image/png")
                except Exception:
                    st.markdown(f"[打开原图 {i}]({url})")
    except Exception as e:
        st.error(f"图片生成失败：{e}")

# ------------------------
# 视频生成（Replicate + 演示模式）
# ------------------------
st.subheader("视频生成（Beta）")
vid_prompt = st.text_input("视频提示词（简述画面/风格/动作）", "教室里孩子跟读Phonics，镜头推拉，温暖氛围")
vid_seconds = st.slider("时长（秒）", 2, 8, 4)
vid_model_slug = st.text_input("模型（Replicate slug）", "stability-ai/stable-video-diffusion")
vid_demo = st.checkbox("演示模式（无API）", value=True, help="默认使用GIF演示")
btn_vid = st.button("生成视频", type="primary")

if btn_vid:
    try:
        if vid_demo:
            st.info("演示模式：生成GIF占位短视频")
            gif_buf = make_placeholder_gif(vid_prompt, frames=max(12, vid_seconds * 6))
            st.image(gif_buf, caption="占位视频（GIF）", use_column_width=True)
            st.download_button("下载GIF", data=gif_buf.getvalue(), file_name="demo.gif", mime="image/gif")
        else:
            outputs = run_replicate_model(vid_model_slug, {
                "prompt": vid_prompt,
                "duration": vid_seconds,
            })
            urls = list(outputs) if not isinstance(outputs, list) else outputs
            if not urls:
                st.warning("未返回视频URL，请更换模型或稍后重试。")
            else:
                vid_url = urls[-1]
                st.video(vid_url)
                st.markdown(f"[下载视频]({vid_url})")
    except Exception as e:
        st.error(f"视频生成失败：{e}")

st.markdown("---")
st.caption("合规提醒：学习效果因人而异，需持续投入；本工具输出仅作参考文案，最终发布前请再次人工审核。")