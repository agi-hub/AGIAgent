"""
AGIAgent 用户数据可视化分析看板
分析 /data 目录下各用户的 manager.out 日志
"""

import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
from pathlib import Path

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AGIAgent 用户分析看板",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 数据加载与解析
# ─────────────────────────────────────────────

def get_data_root() -> str:
    """从侧边栏获取数据根目录（可配置）"""
    _script_dir = Path(os.path.abspath(__file__)).parent
    default = str(_script_dir.parent.parent / "data")
    return st.session_state.get("data_root", default)


@st.cache_data(ttl=30)
def load_all_records(data_root: str) -> pd.DataFrame:
    """
    扫描 data_root 下所有用户目录，解析每个 output_YYYYMMDD_HHMMSS 文件夹的
    manager.out，返回包含以下字段的 DataFrame：
        user        - 用户名
        folder      - output 文件夹名
        date        - 日期 (date 类型)
        datetime    - 完整时间 (datetime 类型)
        requirement - 用户需求字符串（一条记录一行）
    """
    records = []
    root = Path(data_root)
    if not root.exists():
        return pd.DataFrame()

    # 遍历用户目录
    for user_dir in sorted(root.iterdir()):
        if not user_dir.is_dir():
            continue
        user = user_dir.name

        # 遍历 output_YYYYMMDD_HHMMSS 文件夹
        for output_dir in sorted(user_dir.iterdir()):
            if not output_dir.is_dir():
                continue
            m = re.match(r"output_(\d{8})_(\d{6})", output_dir.name)
            if not m:
                continue

            date_str, time_str = m.group(1), m.group(2)
            try:
                dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
            except ValueError:
                continue

            manager_out = output_dir / "logs" / "manager.out"
            requirements = []
            if manager_out.exists():
                try:
                    text = manager_out.read_text(encoding="utf-8", errors="replace")
                    for line in text.splitlines():
                        line = line.strip()
                        if line.startswith("Received user requirement:"):
                            req = line[len("Received user requirement:"):].strip()
                            if req:
                                requirements.append(req)
                except Exception:
                    pass

            if requirements:
                for req in requirements:
                    records.append({
                        "user": user,
                        "folder": output_dir.name,
                        "date": dt.date(),
                        "datetime": dt,
                        "requirement": req,
                    })
            else:
                # 即使没有需求，也记录一条（用于统计访问次数）
                records.append({
                    "user": user,
                    "folder": output_dir.name,
                    "date": dt.date(),
                    "datetime": dt,
                    "requirement": "",
                })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# ─────────────────────────────────────────────
# 侧边栏
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 AGIAgent 分析看板")
    st.markdown("---")

    # 用 os.path.abspath 确保路径在任何工作目录下都能正确解析
    _script_dir = Path(os.path.abspath(__file__)).parent          # .../AGIAgent/dashboard
    default_root = str(_script_dir.parent.parent / "data")        # .../data
    data_root = st.text_input(
        "📂 数据根目录",
        value=default_root,
        help="包含各用户子目录的数据根目录路径",
    )
    st.session_state["data_root"] = data_root

    if st.button("🔄 刷新数据", use_container_width=True):
        st.cache_data.clear()

    st.markdown("---")
    st.markdown(
        """
        **功能说明**
        - 📅 **Tab 1**：按日期查看活跃用户需求
        - 📈 **Tab 2**：用户每日访问次数曲线
        - 👤 **Tab 3**：指定用户的所有需求
        """
    )

# ─────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────

df_all = load_all_records(data_root)

if df_all.empty:
    st.error(f"⚠️ 未找到任何数据，请检查目录是否正确：`{data_root}`")
    st.stop()

all_users = sorted(df_all["user"].unique().tolist())
all_dates = sorted(df_all["date"].dt.date.unique().tolist())

# ─────────────────────────────────────────────
# 主体 Tabs
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📅 按日期查看用户需求", "📈 用户每日访问趋势", "👤 指定用户全部需求", "🏆 用户需求排行榜"]
)

# ═══════════════════════════════════════════════
# Tab 1 ─ 给定日期，列出活跃用户的 user requirements
# ═══════════════════════════════════════════════
with tab1:
    st.header("📅 按日期查看活跃用户需求")
    st.markdown("选择一个日期，查看当天各活跃用户提交的所有用户需求。")

    col_date, col_user_filter = st.columns([2, 2])
    with col_date:
        selected_date = st.date_input(
            "选择日期",
            value=all_dates[-1] if all_dates else date.today(),
            min_value=all_dates[0] if all_dates else date.today(),
            max_value=all_dates[-1] if all_dates else date.today(),
        )
    with col_user_filter:
        user_filter = st.multiselect(
            "筛选用户（留空=全部）",
            options=all_users,
            default=[],
        )

    # 过滤数据
    mask = df_all["date"].dt.date == selected_date
    df_day = df_all[mask].copy()

    if user_filter:
        df_day = df_day[df_day["user"].isin(user_filter)]

    # 只保留有需求的记录
    df_day_req = df_day[df_day["requirement"] != ""].copy()

    if df_day_req.empty:
        st.info(f"📭 {selected_date} 当天没有找到任何用户需求记录。")
    else:
        # 统计摘要
        c1, c2, c3 = st.columns(3)
        c1.metric("活跃用户数", df_day["folder"].nunique() and df_day["user"].nunique())
        c2.metric("会话数（output文件夹）", df_day["folder"].nunique())
        c3.metric("需求条数", len(df_day_req))

        st.markdown("#### 详细需求列表")

        # 构建展示表格
        display_df = df_day_req[["user", "datetime", "folder", "requirement"]].copy()
        display_df = display_df.sort_values(["user", "datetime"])
        display_df.columns = ["用户", "时间", "会话文件夹", "用户需求"]
        display_df["时间"] = display_df["时间"].dt.strftime("%H:%M:%S")

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "用户": st.column_config.TextColumn("用户", width=120),
                "时间": st.column_config.TextColumn("时间", width=90),
                "会话文件夹": st.column_config.TextColumn("会话文件夹", width=220),
                "用户需求": st.column_config.TextColumn("用户需求", width="large"),
            },
        )

        # 按用户分组展示
        st.markdown("#### 按用户分组")
        for user in sorted(df_day_req["user"].unique()):
            user_reqs = df_day_req[df_day_req["user"] == user]
            with st.expander(f"👤 {user}  （{len(user_reqs)} 条需求）", expanded=True):
                for _, row in user_reqs.sort_values("datetime").iterrows():
                    time_str = row["datetime"].strftime("%H:%M:%S")
                    st.markdown(
                        f"🕐 `{time_str}`  📁 `{row['folder']}`\n\n"
                        f"> {row['requirement']}"
                    )

# ═══════════════════════════════════════════════
# Tab 2 ─ 每用户每天访问次数曲线
# ═══════════════════════════════════════════════
with tab2:
    st.header("📈 用户每日访问次数趋势")
    st.markdown(
        "以 `manager.out` 文件（即 output 文件夹）的数量代表每天的访问次数，绘制各用户的趋势曲线。"
    )

    # 去重：每个 folder 只计一次访问
    df_unique = df_all.drop_duplicates(subset=["user", "folder"]).copy()
    df_count = (
        df_unique.groupby(["user", df_unique["date"].dt.date])
        .size()
        .reset_index(name="访问次数")
    )
    df_count.columns = ["用户", "日期", "访问次数"]
    df_count["日期"] = pd.to_datetime(df_count["日期"])

    col_left, col_right = st.columns([3, 1])
    with col_right:
        selected_users_line = st.multiselect(
            "选择显示的用户",
            options=all_users,
            default=all_users,
            key="line_users",
        )
        show_markers = st.checkbox("显示数据点", value=True)
        show_area = st.checkbox("面积填充", value=False)

    with col_left:
        df_plot = df_count[df_count["用户"].isin(selected_users_line)] if selected_users_line else df_count

        if df_plot.empty:
            st.info("没有可显示的数据。")
        else:
            chart_func = px.area if show_area else px.line
            fig = chart_func(
                df_plot,
                x="日期",
                y="访问次数",
                color="用户",
                markers=show_markers,
                title="各用户每日访问次数",
                labels={"日期": "日期", "访问次数": "访问次数（sessions）", "用户": "用户"},
                template="plotly_white",
            )
            fig.update_layout(
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
            )
            fig.update_xaxes(tickformat="%Y-%m-%d", dtick="D1")
            st.plotly_chart(fig, use_container_width=True)

    # 汇总表格
    st.markdown("#### 每日访问次数汇总表")
    if not df_count.empty:
        pivot = df_count.pivot_table(
            index="日期", columns="用户", values="访问次数", fill_value=0
        ).reset_index()
        pivot["日期"] = pivot["日期"].dt.strftime("%Y-%m-%d")
        pivot.columns.name = None
        st.dataframe(pivot, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════
# Tab 3 ─ 指定用户的所有 user requirements
# ═══════════════════════════════════════════════
with tab3:
    st.header("👤 指定用户的所有需求")
    st.markdown("选择一个用户，查看该用户历史上所有的用户需求。")

    col_sel, col_search = st.columns([2, 2])
    with col_sel:
        selected_user = st.selectbox("选择用户", options=all_users)
    with col_search:
        keyword = st.text_input("🔍 关键词过滤（可选）", placeholder="输入关键词筛选需求...")

    df_user = df_all[(df_all["user"] == selected_user) & (df_all["requirement"] != "")].copy()

    if keyword:
        df_user = df_user[df_user["requirement"].str.contains(keyword, case=False, na=False)]

    if df_user.empty:
        st.info(f"📭 用户 **{selected_user}** 没有找到任何需求记录。")
    else:
        df_user = df_user.sort_values("datetime")

        # 统计摘要
        c1, c2, c3 = st.columns(3)
        c1.metric("总需求条数", len(df_user))
        c2.metric("活跃天数", df_user["date"].nunique())
        c3.metric("会话总数", df_user["folder"].nunique())

        st.markdown("#### 需求列表")

        display_user = df_user[["datetime", "folder", "requirement"]].copy()
        display_user.columns = ["时间", "会话文件夹", "用户需求"]
        display_user["时间"] = display_user["时间"].dt.strftime("%Y-%m-%d %H:%M:%S")

        st.dataframe(
            display_user,
            use_container_width=True,
            hide_index=True,
            column_config={
                "时间": st.column_config.TextColumn("时间", width=160),
                "会话文件夹": st.column_config.TextColumn("会话文件夹", width=220),
                "用户需求": st.column_config.TextColumn("用户需求", width="large"),
            },
        )

        # 词频分析
        st.markdown("#### 需求词云 / 词频统计")
        all_req_text = " ".join(df_user["requirement"].tolist())
        # 简单词频（按空格/标点分割中英文词）
        import re as _re
        words = _re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z]{3,}", all_req_text)
        if words:
            from collections import Counter
            word_freq = Counter(words).most_common(20)
            wf_df = pd.DataFrame(word_freq, columns=["词语", "出现次数"])
            fig_bar = px.bar(
                wf_df,
                x="词语",
                y="出现次数",
                title=f"用户 {selected_user} 高频需求词（Top 20）",
                template="plotly_white",
                color="出现次数",
                color_continuous_scale="Blues",
            )
            fig_bar.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("词频分析无结果（需求内容较短）。")

# ═══════════════════════════════════════════════
# Tab 4 ─ 用户需求排行榜（所有日期，按总数排序）
# ═══════════════════════════════════════════════
with tab4:
    st.header("🏆 用户需求排行榜")
    st.markdown("统计所有日期内各用户的需求总数，按需求数量降序排列，每行显示一个用户及其全部需求。")

    df_req_all = df_all[df_all["requirement"] != ""].copy()

    if df_req_all.empty:
        st.info("没有找到任何需求记录。")
    else:
        # 按用户统计总数，降序
        user_counts = (
            df_req_all.groupby("user")
            .size()
            .reset_index(name="需求总数")
            .sort_values("需求总数", ascending=False)
            .reset_index(drop=True)
        )
        user_counts.index += 1  # 排名从 1 开始

        # 顶部摘要指标
        c1, c2, c3 = st.columns(3)
        c1.metric("用户总数", len(user_counts))
        c2.metric("需求总条数", int(user_counts["需求总数"].sum()))
        c3.metric("人均需求数", f"{user_counts['需求总数'].mean():.1f}")

        # 水平柱状图
        fig_rank = px.bar(
            user_counts,
            x="需求总数",
            y="user",
            orientation="h",
            text="需求总数",
            title="各用户需求总数排行",
            template="plotly_white",
            color="需求总数",
            color_continuous_scale="Blues",
        )
        fig_rank.update_layout(
            yaxis=dict(autorange="reversed"),
            height=max(200, len(user_counts) * 50 + 80),
            showlegend=False,
        )
        fig_rank.update_traces(textposition="outside")
        st.plotly_chart(fig_rank, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 各用户详细需求列表")

        # 遍历排名后的用户，每人展示一个 expander
        for rank, row in user_counts.iterrows():
            user = row["user"]
            total = int(row["需求总数"])

            user_reqs = (
                df_req_all[df_req_all["user"] == user]
                .sort_values("datetime")[["datetime", "folder", "requirement"]]
            )

            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🔹")
            with st.expander(f"{medal} #{rank}  **{user}**  —  共 {total} 条需求", expanded=False):
                # 紧凑表格
                display = user_reqs.copy()
                display["datetime"] = display["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
                display.columns = ["时间", "会话文件夹", "用户需求"]
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "时间": st.column_config.TextColumn("时间", width=160),
                        "会话文件夹": st.column_config.TextColumn("会话文件夹", width=220),
                        "用户需求": st.column_config.TextColumn("用户需求", width="large"),
                    },
                )

# ─────────────────────────────────────────────
# 页脚
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(f"数据来源：`{data_root}` ｜ 共加载 {df_all['folder'].nunique()} 个会话，{len(df_all[df_all['requirement'] != ''])} 条需求记录")
