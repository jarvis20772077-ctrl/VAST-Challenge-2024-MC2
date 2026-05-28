import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. 全局配置与汉化字典
# ==========================================
st.set_page_config(page_title="VAST MC2 猎鹰情报系统 ", layout="wide", initial_sidebar_state="expanded")

# ⚠️ 请确保你的 JSON 数据文件存放在此路径下
DATA_DIR = r"E:\code-bit-lesson\Data Visualization\MC2"

TRANSLATION_MAP = {
    # 实体类型
    "Entity.Commodity.Fish": "🐟 鱼类商品",
    "Entity.Location.City": "🏙️ 城市港口",
    "Entity.Location.Point": "📍 海域坐标",
    "Entity.Organization.FishingCompany": "🏢 捕捞公司",
    "Entity.Organization.LogisticsCompany": "🚛 物流公司",
    "Entity.Organization.FinancialCompany": "🏦 金融机构",
    "Entity.Organization.NGO": "🛡️ 非政府组织",
    "Entity.Person": "👤 个人",
    "Entity.Person.CEO": "👔 首席执行官",
    "Entity.Vessel": "🚢 商业船只",

    # 幕后黑手终极汉化
    "SouthSeafood Express Corp": "🏢 SSEC (南大洋海鲜总公司 - 核心黑手)",
    "SSEC": "🏢 SSEC (南大洋海鲜总公司 - 核心黑手)",

    # 事件类型
    "Event.TransportEvent.TransponderPing": "📍 轨迹打卡",
    "Event.Ownership": "🤝 资产所有权",
    "Event.Transaction": "💰 商业交易",
    "Event.Employment": "💼 雇佣关系",
    "Event.FamilyRelationship": "👨‍👩‍👧 家族关系",

    # 关键地名
    "City of Haacklee": "哈克利市 (合法港口)",
    "Oceanus Ecological Preserve": "⚠️ Oceanus 保护区 (禁区)",
    "Wobblego Ecological Preserve": "⚠️ Wobblego 保护区 (禁区)",
    "Donkerbaai": "唐克湾 (Donkerbaai)",
    "Port of Lomian": "洛米安港 (Lomian)"
}

COLOR_MAP = {
    "哈克利市 (合法港口)": "#3498DB",
    "⚠️ Oceanus 保护区 (禁区)": "#E74C3C",
    "⚠️ Wobblego 保护区 (禁区)": "#E74C3C",
    "唐克湾 (Donkerbaai)": "#2ECC71",
    "洛米安港 (Lomian)": "#1ABC9C",
    "未知海域 (信号丢失)": "#7F8C8D"
}


# ==========================================
# 2. 数据加载与预处理核心引擎
# ==========================================
@st.cache_data
def load_and_process_data():
    file_path = os.path.join(DATA_DIR, "mc2.json")
    if not os.path.exists(file_path):
        return pd.DataFrame(), pd.DataFrame()

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes_df = pd.DataFrame(data.get('nodes', []))
    links_df = pd.DataFrame(data.get('links', []))

    if links_df.empty: return nodes_df, links_df

    # 执行全方位去空格与汉化
    for col in ['type', 'source', 'target']:
        if col in links_df.columns:
            links_df[col] = links_df[col].astype(str).str.strip().replace(TRANSLATION_MAP)

    if not nodes_df.empty and 'type' in nodes_df.columns:
        nodes_df['type'] = nodes_df['type'].astype(str).str.strip().replace(TRANSLATION_MAP)

    # 时序特征构建
    if 'time' in links_df.columns:
        links_df['time'] = pd.to_datetime(links_df['time'], errors='coerce')
        links_df = links_df.sort_values(by=['target', 'time']).reset_index(drop=True)

    if 'dwell' in links_df.columns:
        links_df['end_time'] = links_df['time'] + pd.to_timedelta(links_df['dwell'].fillna(0), unit='s')

    # 计算信号黑洞(Gap)，这是判定非法捕捞的关键
    ping_mask = links_df['type'] == "📍 轨迹打卡"
    links_df.loc[ping_mask, 'time_diff'] = links_df[ping_mask].groupby('target')['time'].diff().dt.total_seconds()

    return nodes_df, links_df


# ==========================================
# 3. 智能嫌疑打分算法
# ==========================================
def calculate_suspicion_scores(links_df):
    ping_df = links_df[links_df['type'] == "📍 轨迹打卡"].copy()
    if ping_df.empty: return pd.DataFrame()

    scores = []
    for vessel, group in ping_df.groupby('target'):
        illegal_dwell = group[group['source'].str.contains("禁区", na=False)]['dwell'].sum() / 3600
        gap_count = (group['time_diff'] > 14400).sum()
        total_score = (illegal_dwell * 2.5) + (gap_count * 10)

        if total_score > 0:
            scores.append({"船只ID": vessel, "非法停留(h)": round(illegal_dwell, 1),
                           "异常断联(次)": gap_count, "总嫌疑分": round(total_score, 1)})

    return pd.DataFrame(scores).sort_values(by="总嫌疑分", ascending=False) if scores else pd.DataFrame()


# ==========================================
# 4. 可视化侦查组件
# ==========================================
def render_business_network(links_df, target_vessel):
    """提取该船只的幕后老板，追踪资金流向代替名义所有权 (带根节点强化视觉)"""
    biz_types = ["🤝 资产所有权", "💼 雇佣关系", "👨‍👩‍👧 家族关系", "💰 商业交易"]
    biz_links = links_df[(links_df['type'].isin(biz_types)) &
                         ((links_df['target'] == target_vessel) | (links_df['source'] == target_vessel))].copy()

    if biz_links.empty:
        st.error("🚨 【高危警报：幽灵船特征】")
        st.warning("该船只不仅没有合法的『资产所有权』登记，甚至没有任何资金记录。建议立刻结合『物流溯源』进行跨表追踪！")
        return

    def get_biz_partner(row):
        return row['source'] if row['target'] == target_vessel else row['target']

    biz_links['关联方'] = biz_links.apply(get_biz_partner, axis=1)
    biz_links = biz_links.dropna(subset=['关联方'])
    biz_links = biz_links[(biz_links['关联方'] != 'nan') & (biz_links['关联方'] != '')]

    if biz_links.empty: return

    # 绘制带固定根节点的商业穿透树状图
    fig_tree = px.treemap(
        biz_links,
        path=[px.Constant(f"🎯 侦查原点: {target_vessel}"), 'type', '关联方'],
        title="🏢 实际控制权与资金流向穿透",
        color='type',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_tree.update_traces(textinfo="label+value", textfont=dict(size=14), root_color="#F2F3F4")
    fig_tree.update_layout(height=450, margin=dict(t=40, l=10, r=10, b=10))
    st.plotly_chart(fig_tree, use_container_width=True)


def render_entity_profile(links_df, target_vessel):
    """高度结构化的实体接触画像分析 (解决NaN崩溃问题)"""
    related = links_df[(links_df['target'] == target_vessel) | (links_df['source'] == target_vessel)].copy()
    if related.empty: return

    related['关联实体'] = related.apply(lambda r: r['target'] if r['source'] == target_vessel else r['source'], axis=1)

    # 清洗：严格剔除空值
    profile_df = related.groupby(['type', '关联实体']).size().reset_index(name='频次')
    profile_df = profile_df.dropna(subset=['type', '关联实体'])
    profile_df = profile_df[(profile_df['关联实体'] != "nan") & (profile_df['关联实体'] != "")]

    if profile_df.empty: return

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_sun = px.sunburst(profile_df, path=['type', '关联实体'], values='频次', color='type',
                              title="🎯 行为关系结构穿透", color_discrete_sequence=px.colors.qualitative.Safe)
        fig_sun.update_layout(height=450, margin=dict(t=40, l=0, r=0, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)

    with col2:
        top_entities = profile_df.groupby('关联实体')['频次'].sum().reset_index().sort_values(by='频次',
                                                                                              ascending=False).head(10)
        fig_bar = px.bar(top_entities, x='频次', y='关联实体', orientation='h', title="📊 紧密接触对象排行",
                         color='频次', color_continuous_scale="Reds")
        fig_bar.update_layout(height=450, yaxis={'categoryorder': 'total ascending'},
                              margin=dict(t=40, l=10, r=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)


def render_sankey(links_df, target_vessel):
    """警示性推断物流溯源"""
    vessel_ports = links_df[(links_df['target'] == target_vessel) & (links_df['type'] == '📍 轨迹打卡')]
    if vessel_ports.empty: return

    all_nodes = list(
        set([target_vessel] + vessel_ports['source'].dropna().tolist() + ["🐟 疑似非法黑产", "🐟 合法商业渔获"]))
    node_indices = {name: i for i, name in enumerate(all_nodes)}

    src_idx, tgt_idx, vals, colors = [], [], [], []
    for port, count in vessel_ports['source'].value_counts().items():
        if str(port) == "nan" or pd.isna(port): continue
        is_danger = "禁区" in str(port)
        c = "rgba(231, 76, 60, 0.45)" if is_danger else "rgba(52, 152, 219, 0.3)"

        src_idx.extend([node_indices[target_vessel], node_indices[port]])
        tgt_idx.extend([node_indices[port], node_indices["🐟 疑似非法黑产" if is_danger else "🐟 合法商业渔获"]])
        vals.extend([count * 10, count * 8])  # 放大数值以优化视觉
        colors.extend([c, c])

    fig_sk = go.Figure(data=[go.Sankey(node=dict(pad=40, thickness=20, label=all_nodes, color="#D5D8DC"),
                                       link=dict(source=src_idx, target=tgt_idx, value=vals, color=colors))])
    fig_sk.update_layout(height=450, margin=dict(t=20, l=10, r=10, b=10), font=dict(size=13))
    st.plotly_chart(fig_sk, use_container_width=True)


# ==========================================
# 5. 系统主入口与布局
# ==========================================
def main():
    st.title("🚢 VAST MC2: 猎鹰 (Falcon) 全维度情报系统")

    nodes_df, links_df = load_and_process_data()
    if links_df.empty:
        st.error("无法加载数据，请检查 E:\\code-bit-lesson\\Data Visualization\\MC2\\mc2.json 是否存在。")
        st.stop()

    # ================= 右侧监控大屏 =================
    with st.sidebar:
        st.header("🚨 智能嫌疑追踪雷达")
        st.markdown("系统基于『禁区非法驻留时长』与『信号蓄意丢失』智能打分：")

        scores_df = calculate_suspicion_scores(links_df)
        default_suspect = None
        if not scores_df.empty:
            st.dataframe(scores_df.head(10), hide_index=True, use_container_width=True)
            default_suspect = scores_df.iloc[0]['船只ID']
        else:
            st.success("✅ 当前未发现高危嫌疑目标。")

        st.markdown("---")
        st.header("🎯 目标档案提取")
        all_vessels = links_df[links_df['type'] == "📍 轨迹打卡"]['target'].dropna().unique()
        default_idx = list(all_vessels).index(default_suspect) if default_suspect in all_vessels else 0
        selected_vessel = st.selectbox("监控目标锁定：", all_vessels, index=default_idx)

        # 物理档案属性提取
        st.subheader("📋 目标物理特征 (Node Info)")
        if not nodes_df.empty and 'id' in nodes_df.columns:
            vessel_node = nodes_df[nodes_df['id'] == selected_vessel]
            if not vessel_node.empty:
                v_info = vessel_node.iloc[0]
                st.caption(f"**船名:** {v_info.get('Name', '未登记 (高风险)')}")
                st.caption(f"**注册国:** {v_info.get('flag_country', '未登记 (高风险)')}")
                st.caption(f"**吨位 (Tonnage):** {v_info.get('tonnage', '未知')}")
            else:
                st.caption("⚠️ 节点数据库中查无此船物理档案。")

    # ================= 主视窗 =================
    st.markdown(f"### 🔍 实时侦察窗口: [{selected_vessel}]")
    v_data = links_df[(links_df['target'] == selected_vessel) & (links_df['type'] == '📍 轨迹打卡')].copy()

    if not v_data.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("记录总数", f"{len(v_data)} 次")
        c2.metric("禁区非法入侵", f"{v_data['source'].str.contains('禁区', na=False).sum()} 次", delta="重点关注",
                  delta_color="inverse")
        c3.metric("异常断联/黑洞", f"{(v_data['time_diff'] > 14400).sum()} 次", delta="逃避监管", delta_color="inverse")

        st.markdown("<br>", unsafe_allow_html=True)
        # 赋予颜色语义的甘特图
        v_data['状态颜色'] = v_data['source'].apply(
            lambda x: COLOR_MAP.get(x, "#BDC3C7") if isinstance(x, str) else "#BDC3C7")
        fig_gantt = px.timeline(v_data, x_start="time", x_end="end_time", y="source",
                                color="source", color_discrete_map=COLOR_MAP, height=280)
        fig_gantt.update_layout(showlegend=False, margin=dict(t=0, b=10, l=10, r=10), yaxis_title="所在海域/港口")
        st.plotly_chart(fig_gantt, use_container_width=True)

    st.markdown("---")
    # ================= 多维侦查分页 (扩展为5个Tab) =================
    t1, t2, t3, t4, t5 = st.tabs([
        "🏛️ 幕后控制穿透 (Treemap)",
        "📊 实体交往画像 (Sunburst)",
        "🔀 物流黑产溯源 (Sankey)",
        "📄 电子案卷明细 (Data)",
        "🕵️ 数据溯源与防伪排查 (Forensics)"
    ])

    with t1:
        st.subheader("商业资产与资金链穿透")
        st.markdown(
            "💡 **侦查逻辑**：直接剥开『所有权』的伪装，追踪交易流向。如果所有的钱最终都流向特定机构（如SSEC），那它就是幕后黑手。")
        render_business_network(links_df, selected_vessel)

    with t2:
        st.subheader("高频接触实体排查")
        render_entity_profile(links_df, selected_vessel)

    with t3:
        st.subheader("疑似非法渔获洗白推演")
        st.markdown("💡 **侦查逻辑**：将该船只接触过『生态禁区』的轨迹流标注为红色，追踪其随后靠港并销赃的潜在路径。")
        render_sankey(links_df, selected_vessel)

    with t4:
        st.subheader("底层数据过滤 (交易与轨迹)")
        filter_type = st.selectbox("选择要提取的电子案卷类型：", ["全部记录", "💰 商业交易", "📍 轨迹打卡"])
        base_data = links_df[(links_df['target'] == selected_vessel) | (links_df['source'] == selected_vessel)]
        if filter_type != "全部记录":
            base_data = base_data[base_data['type'] == filter_type]
        st.dataframe(base_data.sort_values(by='time', ascending=False), use_container_width=True)

    with t5:
        st.subheader("⚠️ 电子数据取证 (Digital Forensics)")
        st.markdown("💡 **侦查逻辑**：核对元数据（Metadata）真伪，追查是否存在内鬼篡改轨迹数据，或利用异常字段传递非法暗号。")

        # 提取当前船只的所有关联数据
        vessel_all_data = links_df[(links_df['target'] == selected_vessel) | (links_df['source'] == selected_vessel)]

        # 1. 数据库编辑痕迹排查
        st.markdown("#### 👤 数据库编辑痕迹排查")
        if '_last_edited_by' in links_df.columns:
            if not vessel_all_data.empty:
                edit_counts = vessel_all_data['_last_edited_by'].value_counts().reset_index()
                edit_counts.columns = ['最后编辑账号', '修改记录数']

                c_edit1, c_edit2 = st.columns([1, 2])
                with c_edit1:
                    st.dataframe(edit_counts, hide_index=True)
                with c_edit2:
                    st.info(
                        "📌 **审查建议**：正常轨迹数据应由系统（如 OVLS）自动生成。如果发现关键的轨迹断联点是由**特定个人账号**（临时工/黑客）最后编辑的，极大概率是有人在数据库中**手动删除了非法轨迹**来销毁证据。")
        else:
            st.warning("当前数据集切片中未提取到编辑者元数据 (`_last_edited_by`)。")

        st.markdown("---")

        # 2. 异常通讯字段审查
        st.markdown("#### 📡 隐秘暗号与通讯排查")
        suspicious_cols = [col for col in links_df.columns if
                           col in ['aphorism', 'holiday_greeting', 'wisdom', 'saying of the sea']]

        if suspicious_cols and not vessel_all_data.empty:
            st.write("在底层数据中检测到非标准业务字段，疑似隐秘通讯通道：")
            # 筛选出这些列不为空的记录
            secret_msgs = vessel_all_data.dropna(subset=suspicious_cols, how='all')
            if not secret_msgs.empty:
                st.error(
                    "🚨 发现隐藏信息！这些看似无关紧要的格言或节日问候，极可能是非法渔船在公海接头、转运货物的暗号。请重点比对这些信息发送的时间点。")
                st.dataframe(secret_msgs[['time', 'type', 'source'] + suspicious_cols], use_container_width=True)
            else:
                st.success("✅ 该船只相关记录中暂未发现隐含的文本暗号。")
        else:
            st.write("数据中未暴露异常文本字段（或该实体未触发此规则）。")


if __name__ == "__main__":
    main()