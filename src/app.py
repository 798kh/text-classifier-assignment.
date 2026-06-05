"""
电商评论情感分析系统
====================
使用 Jieba 分词 + TF-IDF 特征提取 + MultinomialNB 朴素贝叶斯分类器，
对用户输入的中文电商评论进行好评/差评二分类判断。

技术栈: Python + Streamlit + Scikit-learn + Jieba + Pandas
"""

import os
import sys
from pathlib import Path

# --- 兼容本地 .local_packages 目录（解决部分机器 pip install 权限不足的问题）---
_LOCAL_PACKAGES = Path(__file__).parent.parent / ".local_packages"
if _LOCAL_PACKAGES.exists():
    sys.path.insert(0, str(_LOCAL_PACKAGES))

import pandas as pd
import jieba
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# ============================================================================
# 页面配置 — 必须放在所有 Streamlit 命令的最前面
# ============================================================================
st.set_page_config(
    page_title="电商评论情感分析系统",
    page_icon="🛒",
    layout="centered",
)


# ============================================================================
# 模型训练模块（使用缓存装饰器，避免每次交互都重新训练）
# ============================================================================
@st.cache_resource
def load_and_train_model():
    """
    加载数据集，使用 Jieba 分词 + TF-IDF 特征提取，
    训练 MultinomialNB 朴素贝叶斯分类器。

    返回:
        vectorizer: 训练好的 TF-IDF 向量化器
        model:      训练好的 MultinomialNB 分类器
        df:         原始数据集 DataFrame（用于侧边栏展示）
    """
    # --- 1. 加载数据集（使用相对路径，兼容本地运行和 Streamlit Cloud 部署）---
    data_path = Path(__file__).parent.parent / "data.csv"
    df = pd.read_csv(data_path)

    # --- 2. 使用 Jieba 对每条评论文本进行中文分词 ---
    df["tokens"] = df["text"].apply(lambda x: " ".join(jieba.cut(str(x))))

    # --- 3. TF-IDF 特征提取 ---
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["tokens"])

    # --- 4. 训练 MultinomialNB 朴素贝叶斯分类器 ---
    y = df["label"]
    model = MultinomialNB()
    model.fit(X, y)

    return vectorizer, model, df


# ============================================================================
# 加载模型（首次调用会训练，后续调用直接使用缓存）
# ============================================================================
vectorizer, model, df = load_and_train_model()


# ============================================================================
# 侧边栏 — 展示模型状态与数据集信息
# ============================================================================
with st.sidebar:
    st.header("📊 系统状态")
    st.success("✅ 模型已就绪")

    st.metric("训练样本数", len(df))
    st.metric("好评数量", int(df["label"].sum()))
    st.metric("差评数量", int(len(df) - df["label"].sum()))

    st.divider()
    st.caption("📌 算法: Jieba 分词 + TF-IDF + MultinomialNB")
    st.caption("🛒 适用场景: 电商评论情感二分类")

    # 展开查看训练数据集
    with st.expander("📋 查看训练数据集"):
        st.dataframe(df[["text", "label"]], use_container_width=True)


# ============================================================================
# 主页面 — UI 布局
# ============================================================================

# --- 网页标题与系统介绍 ---
st.title("🛒 电商评论情感分析系统")
st.markdown(
    """
    欢迎使用 **电商评论情感分析系统**！本系统基于 **朴素贝叶斯分类器**，
    结合 **Jieba 中文分词** 与 **TF-IDF 特征提取** 技术，
    能够智能识别电商评论的**情感倾向**（好评 / 差评）。

    ---
    **🔍 使用方法：** 在下方文本框中输入任意中文商品评论，
    点击 **"开始 AI 智能分析"** 按钮，系统即可自动判断情感倾向。
    """
)

st.divider()

# --- 用户输入区域 ---
st.subheader("💬 请输入待分析的评论内容")

user_input = st.text_area(
    label="评论内容",
    placeholder="例如：这款手机运行速度超快，拍照效果惊艳，非常满意！",
    height=150,
    label_visibility="collapsed",
)

# --- 分析按钮 ---
analyze_btn = st.button(
    label="🤖 开始 AI 智能分析",
    type="primary",
    use_container_width=True,
)


# ============================================================================
# 预测与结果展示
# ============================================================================
if analyze_btn:
    # 空输入校验
    if not user_input.strip():
        st.warning("⚠️ 请输入评论内容后再进行分析！")
    else:
        # --- 1. 对用户输入进行 Jieba 分词 ---
        user_tokens = " ".join(jieba.cut(user_input.strip()))

        # --- 2. 使用训练好的 TF-IDF 向量化器进行特征转换 ---
        user_vec = vectorizer.transform([user_tokens])

        # --- 3. 模型预测 ---
        prediction = model.predict(user_vec)[0]           # 0 或 1
        proba = model.predict_proba(user_vec)[0]          # [差评概率, 好评概率]

        # --- 4. 显示预测结果 ---
        st.divider()
        st.subheader("📈 分析结果")

        if prediction == 1:
            # 好评 — 绿色横幅 + 气球动画
            st.success(f"🎉 该评论被判定为：**好评**  （置信度: {proba[1]:.2%}）")
            st.balloons()
        else:
            # 差评 — 红色横幅
            st.error(f"😞 该评论被判定为：**差评**  （置信度: {proba[0]:.2%}）")

        # --- 5. 概率柱状图 ---
        st.subheader("📊 概率得分详情")
        proba_df = pd.DataFrame(
            {
                "情感类别": ["差评 (0)", "好评 (1)"],
                "概率得分": [proba[0], proba[1]],
            }
        ).set_index("情感类别")
        st.bar_chart(proba_df, use_container_width=True)

        # --- 6. 分词结果展示（调试用，折叠区） ---
        with st.expander("🔍 查看分词结果"):
            st.code(user_tokens, language="text")
