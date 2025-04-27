import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

# --- 环境配置 ---
load_dotenv()


# --- 数据库连接 ---
@st.cache_resource
def get_db_engine():
    return create_engine(
        f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


# --- 数据加载 ---
@st.cache_data(ttl=3600)
def load_data():
    engine = get_db_engine()
    return pd.read_sql_table("merged_education_data", engine)


# --- 页面配置 ---
st.set_page_config(
    page_title="数据导向的教育决策多元动态洞察系统",
    page_icon="📊",
    layout="wide"
)

# --- 指标中文映射 ---
INDICATOR_MAPPING = {
    # 学生相关
    "reading_score": "阅读成绩",
    "literary_purpose": "文学体验目的",
    "info_purpose": "信息获取目的",
    "integration_process": "理解整合能力",
    "inference_process": "检索推理能力",
    "reading_level": "阅读量级",
    "weekly_reading_hours": "每周阅读时长",
    "interest_reading_freq": "兴趣阅读频率",
    "reading_time_outside": "课外阅读时间",
    # 家庭背景
    "home_books": "家庭藏书量",
    "children_book_count": "儿童书籍数量",
    "study_space_count": "家庭学习空间数",
    "guardian_a_education": "监护人A教育程度",
    "guardian_b_education": "监护人B教育程度",
    "child_education_expect": "孩子教育期望",
    # 学校资源
    "teaching_days_per_year": "每年教学天数",
    "teaching_hours_per_week": "每周教学时长",
    "computer_count": "学校计算机数量",
    "class_library_books": "班级图书馆藏书量",
    # 教师背景
    "teaching_years": "教师教龄",
    "provide_materials_freq": "教材提供频率",
    "encourage_comprehension_freq": "鼓励理解频率"
}


# --- 主程序 ---
def main():
    st.title("📊 数据导向的教育决策多元动态洞察系统")
    df = load_data()

    # --- 侧边栏控件 ---
    with st.sidebar:
        st.header("分析设置")

        # 1. X/Y轴指标选择
        x_axis = st.selectbox(
            "选择X轴指标",
            options=list(INDICATOR_MAPPING.keys()),
            format_func=lambda x: INDICATOR_MAPPING[x]
        )

        y_axes = st.multiselect(
            "选择Y轴指标（可多选）",
            options=list(INDICATOR_MAPPING.keys()),
            format_func=lambda x: INDICATOR_MAPPING[x],
            default=["teaching_hours_per_week"]
        )

        # 2. 数据筛选
        st.subheader("数据筛选")
        selected_countries = st.multiselect(
            "选择国家/地区",
            options=df["IDCNTRY"].unique()
        )

    # --- 数据预处理 ---
    # 应用国家筛选
    if selected_countries:
        df = df[df["IDCNTRY"].isin(selected_countries)]

    # 按国家分组聚合
    group_by = "IDCNTRY"  # 固定分组维度为国家
    agg_df = df.groupby(group_by).agg(
        **{f"avg_{x_axis}": (x_axis, "mean")},
        **{f"avg_{y}": (y, "mean") for y in y_axes}
    ).reset_index()

    # --- 可视化模块 ---
    st.subheader(f"国家/地区维度分析：{INDICATOR_MAPPING[x_axis]} vs {'/'.join([INDICATOR_MAPPING[y] for y in y_axes])}")

    # 动态生成图表
    fig = px.scatter(
        agg_df,
        x=f"avg_{x_axis}",
        y=[f"avg_{y}" for y in y_axes],
        color=group_by,
        hover_name=group_by,
        labels={
            "value": "指标平均值",
            "variable": "指标类型",
            f"avg_{x_axis}": f"平均 {INDICATOR_MAPPING[x_axis]}",
            "IDCNTRY": "国家/地区"
        },
        height=600
    )

    # 多Y轴处理
    if len(y_axes) > 1:
        for i, y in enumerate(y_axes[1:]):
            fig.add_scatter(
                x=agg_df[f"avg_{x_axis}"],
                y=agg_df[f"avg_{y}"],
                name=INDICATOR_MAPPING[y],
                mode="markers",
                marker=dict(symbol=i + 2, size=12),
                yaxis=f"y{i + 2}"
            )
        fig.update_layout(
            yaxis2=dict(
                title=INDICATOR_MAPPING[y_axes[1]],
                overlaying="y",
                side="right"
            )
        )

    # 图表样式优化
    fig.update_layout(
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(title="国家", orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title=f"平均 {INDICATOR_MAPPING[x_axis]}", showgrid=True),
        yaxis=dict(title=INDICATOR_MAPPING[y_axes[0]], showgrid=True)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- 数据展示 ---
    with st.expander("查看原始数据"):
        # 仅对数值列应用格式化
        styled_df = agg_df.style.format(
            "{:.2f}",
            subset=pd.IndexSlice[:, agg_df.select_dtypes(include='number').columns]
        )
        st.dataframe(styled_df, height=300)

    # --- 数据下载 ---
    st.download_button(
        label="📥 下载分析结果",
        data=agg_df.to_csv(index=False).encode("utf-8"),
        file_name="country_analysis.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()