import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
import re
from collections import Counter
import pandas as pd
from pyecharts.charts import WordCloud, Bar, Line, Pie, Radar, Scatter, Funnel, Gauge
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from streamlit_echarts import st_pyecharts  # Streamlit集成pyecharts

# ---------------------- 1. 全局配置 ----------------------
# 页面基础设置
st.set_page_config(
    page_title="URL文章分词可视化工具",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载停用词表
def load_stopwords():
    """加载停用词（优先本地文件，无则用默认集合）"""
    default_stopwords = {
        "的", "了", "是", "我", "你", "他", "她", "它", "们", "在", "和", "与", "或",
        "就", "都", "而", "及", "即", "也", "又", "还", "因", "为", "以", "于", "之",
        "这", "那", "此", "彼", "个", "些", "能", "可", "会", "应", "要", "将", "把",
        "对", "对于", "关于", "通过", "随着", "按照", "基于", "根据", "如果", "假如"
    }
    try:
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            return set([line.strip() for line in f if line.strip()])
    except FileNotFoundError:
        st.warning("未找到停用词文件，使用默认停用词表")
        return default_stopwords

STOPWORDS = load_stopwords()

# ---------------------- 2. 核心功能函数 ----------------------
def crawl_url_article(url: str) -> tuple:
    """爬取URL文章正文"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding  # 自动识别编码
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 移除无关标签
        for tag in soup(["script", "style", "nav", "footer", "aside", "header"]):
            tag.decompose()
        
        # 提取正文（适配新闻/博客页面）
        content_tags = soup.find_all("article") or soup.find_all("div", class_=lambda x: x and ("content" in x.lower() or "article" in x.lower())) or soup.find_all("p")
        article_text = "\n".join([tag.get_text().strip() for tag in content_tags if tag.get_text().strip()])
        
        if not article_text:
            return None, "未提取到正文（可能是动态页面/标签不匹配）"
        return article_text, ""
    except Exception as e:
        return None, f"爬取失败：{str(e)}"

def clean_and_segment(text: str) -> tuple:
    """文本清洗+分词+词频统计"""
    # 清洗文本
    text = re.sub(r"<[^>]+>", "", text)  # 移除HTML标签
    text = re.sub(r"[0-9a-zA-Z\s+]", "", text)  # 移除数字/字母/多余空格
    text = re.sub(r"[^\u4e00-\u9fa5，。！？；：、（）【】]", "", text)  # 仅保留中文
    
    # 分词+过滤停用词/单字
    seg_list = jieba.lcut(text)
    seg_list = [word for word in seg_list if word not in STOPWORDS and len(word) > 1 and word.strip()]
    
    # 词频统计
    word_count = Counter(seg_list)
    return seg_list, word_count

def filter_low_freq_words(word_count: Counter, min_freq: int) -> Counter:
    """过滤低频词"""
    return Counter({word: count for word, count in word_count.items() if count >= min_freq})

# ---------------------- 3. 图表生成函数 ----------------------
def generate_chart(chart_type: str, word_data: list):
    """根据选择的图表类型生成Pyecharts图表"""
    # 取TOP20数据
    top20_data = word_data[:20]
    words = [item[0] for item in top20_data]
    counts = [item[1] for item in top20_data]
    
    if chart_type == "词云":
        c = (
            WordCloud(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add("", top20_data, word_size_range=[20, 100])
            .set_global_opts(title_opts=opts.TitleOpts(title="词频TOP20词云图", subtitle="过滤低频词后"))
        )
    elif chart_type == "柱状图":
        c = (
            Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add_xaxis(words)
            .add_yaxis("词频", counts)
            .reversal_axis()  # 横向柱状图（适配长文本）
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频TOP20柱状图"),
                xaxis_opts=opts.AxisOpts(name="词频"),
                yaxis_opts=opts.AxisOpts(name="词汇")
            )
        )
    elif chart_type == "折线图":
        c = (
            Line(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add_xaxis(words)
            .add_yaxis("词频", counts, markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max"), opts.MarkPointItem(type_="min")]))
            .set_global_opts(title_opts=opts.TitleOpts(title="词频TOP20折线图"))
        )
    elif chart_type == "饼图":
        c = (
            Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add("", top20_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="词频TOP20饼图"), legend_opts=opts.LegendOpts(orient="vertical", pos_top="10%", pos_left="80%"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )
    elif chart_type == "雷达图":
        c = (
            Radar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add_schema(schema=[opts.RadarIndicatorItem(name=word, max_=max(counts)) for word in words[:10]])  # 仅展示前10个（避免雷达图过密）
            .add("词频", [counts[:10]])
            .set_global_opts(title_opts=opts.TitleOpts(title="词频TOP10雷达图"))
        )
    elif chart_type == "散点图":
        c = (
            Scatter(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add_xaxis(words)
            .add_yaxis("词频", counts)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="词频TOP20散点图"),
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
                yaxis_opts=opts.AxisOpts(name="词频")
            )
        )
    elif chart_type == "漏斗图":
        c = (
            Funnel(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add("", top20_data)
            .set_global_opts(title_opts=opts.TitleOpts(title="词频TOP20漏斗图"))
            .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
        )
    elif chart_type == "仪表盘":
        # 仪表盘展示TOP1词汇的词频（适配单值展示）
        top1_word, top1_count = top20_data[0] if top20_data else ("无数据", 0)
        c = (
            Gauge(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="100%", height="600px"))
            .add(f"词频", [(top1_word, top1_count)])
            .set_global_opts(
                title_opts=opts.TitleOpts(title=f"高频词TOP1：{top1_word}"),
                legend_opts=opts.LegendOpts(is_show=False)
            )
        )
    return c

# ---------------------- 4. Streamlit页面布局 ----------------------
def main():
    st.title("📝 URL文章分词可视化分析工具")
    st.markdown("---")

    # 左侧边栏（图表筛选+参数配置）
    with st.sidebar:
        st.header("🔧 图表筛选与配置")
        # 图表类型选择（至少7种）
        chart_type = st.selectbox(
            "选择图表类型",
            ["词云", "柱状图", "折线图", "饼图", "雷达图", "散点图", "漏斗图", "仪表盘"],
            index=0
        )
        # 低频词过滤阈值
        min_freq = st.slider(
            "过滤低频词（最小出现次数）",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="仅展示出现次数≥该值的词汇"
        )
        st.markdown("---")
        st.info("💡 操作说明：输入URL→爬取文章→自动分词→选择图表类型查看结果")

    # 主页面：URL输入+爬取
    col1, col2 = st.columns([3, 1])
    with col1:
        url = st.text_input("📌 输入文章URL", placeholder="例如：https://www.ithome.com/0/780/123.htm")
    with col2:
        crawl_btn = st.button("🚀 爬取并分析", type="primary")

    # 初始化会话状态（保存词频数据，避免重复爬取）
    if "word_count" not in st.session_state:
        st.session_state.word_count = Counter()

    # 爬取+分析逻辑
    if crawl_btn and url:
        with st.spinner("正在爬取文章并分析..."):
            # 1. 爬取文章
            article_text, error = crawl_url_article(url)
            if error:
                st.error(error)
                return
            st.success(f"✅ 文章爬取成功！原始正文长度：{len(article_text)} 字")

            # 2. 清洗分词+词频统计
            seg_list, word_count = clean_and_segment(article_text)
            st.session_state.word_count = word_count
            st.info(f"📊 分词完成！有效分词数：{len(seg_list)} | 唯一词汇数：{len(word_count)}")

    # 展示结果（有词频数据时）
    if st.session_state.word_count:
        st.markdown("---")
        # 过滤低频词
        filtered_word_count = filter_low_freq_words(st.session_state.word_count, min_freq)
        if not filtered_word_count:
            st.warning(f"⚠️ 过滤后无数据（最小词频设为{min_freq}，可降低阈值重试）")
            return
        
        # 排序取TOP20
        sorted_word_data = filtered_word_count.most_common(20)
        
        # 展示词频TOP20表格
        st.subheader("📈 词频排名TOP20（过滤低频词后）")
        top20_df = pd.DataFrame(sorted_word_data, columns=["词汇", "出现次数"])
        st.dataframe(top20_df, use_container_width=True)

        # 生成并展示图表
        st.subheader(f"🎨 {chart_type}展示")
        chart = generate_chart(chart_type, sorted_word_data)
        st_pyecharts(chart, key=chart_type)  # key确保切换图表时重新渲染

if __name__ == "__main__":
    main()