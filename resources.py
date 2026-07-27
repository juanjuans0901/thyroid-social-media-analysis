"""
Shared domain resources — single source of truth.
Mirrors the published supplement of DHJ-26-0922:
  MEDICAL_TERMS      -> Table S1 (jieba user dictionary)
  STOPWORDS          -> Table S2 (stopword list)
  SENTIMENT_LEXICON  -> Table S4 (domain sentiment lexicon, 80-term optimized)
  EMOTION_CATEGORIES -> 7-category DUTIR emotion mapping (Figure S3, Table S12)

STOPWORDS contains the 172 function words listed in Table S2 and is the list that
reproduces the published five-topic solution (T1 = 263, T2 = 206, T3 = 434,
T4 = 552, T5 = 461). Do not substitute another stopword list.
"""

MEDICAL_TERMS = [
    "甲状腺微小乳头状癌", "甲状腺乳头状癌", "甲状腺癌", "甲状腺结节",
    "微小癌", "乳头状癌", "滤泡癌", "髓样癌", "未分化癌",
    "甲状腺全切", "半切", "全切", "消融术", "射频消融", "微波消融",
    "热消融", "细针穿刺", "穿刺活检", "淋巴结清扫", "中央区清扫",
    "优甲乐", "左甲状腺素", "碘131", "碘治疗", "同位素治疗",
    "促甲状腺激素", "TSH", "TG", "甲功", "甲状腺功能",
    "TI-RADS", "TIRADS", "4a", "4b", "4c",
    "B超", "超声", "颈部超声", "增强CT", "病理报告",
    "基因检测", "BRAF", "基因突变",
    "淋巴结转移", "被膜侵犯", "远处转移", "复发",
    "主动监测", "随访观察", "积极监测", "观察等待",
    "内分泌科", "甲乳外科", "头颈外科",
    "三甲医院", "好大夫", "中国医学科学院", "北京协和",
    "术后复查", "终身服药", "抑制治疗",
    "过度治疗", "过度诊断", "惰性癌", "懒癌"
]

STOPWORDS = set([
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "吗", "什么", "没", "那", "还", "做", "想", "能", "但是", "这个",
    "来", "吧", "啊", "呢", "怎么", "可以", "就是", "知道", "如果", "现在",
    "因为", "所以", "但", "已经", "只", "还是", "或者", "比较", "其实", "然后",
    "这样", "那个", "这些", "应该", "可能", "时候", "真的", "最", "让", "把",
    "被", "给", "从", "里", "用", "对", "而", "些", "更", "太",
    "两", "为了", "之后", "过", "个", "下", "后", "中", "多", "大",
    "小", "长", "时间", "比", "再", "又", "才", "等", "第", "次",
    "此", "以", "于", "与", "并", "且", "而且", "不过", "只是", "虽然",
    "以为", "之前", "之间", "为", "地", "得", "最后", "什么样", "哪个", "每",
    "当", "通过", "关于", "其", "其他", "另外", "同时", "除了", "包括", "以及",
    "不是", "不要", "这种", "那些", "它", "它们", "他们", "她们", "我们", "你们",
    "这里", "那里", "如何", "怎样", "非常", "特别", "所有", "每个", "任何", "某",
    "别", "各", "整个", "本", "该", "哪", "谁", "哪里", "啥", "嗯",
    "哦", "嘛", "哈", "呀", "喔", "哎", "唉", "额", "哇", "嘿",
    "噢", "诶"
])

SENTIMENT_LEXICON = {
    # --- 负面词 (47个) ---
    # Weight 3 (强)
    "五雷轰顶": -3, "噩梦": -3, "天塌": -3, "崩溃": -3,
    "恐惧": -3, "恐慌": -3, "悲伤": -3, "愤怒": -3,
    "抑郁": -3, "折磨": -3, "晴天霹雳": -3, "煎熬": -3,
    "痛苦": -3, "绝望": -3,
    # Weight 2 (中)
    "伤心": -2, "厌恶": -2, "后悔": -2, "哭": -2,
    "失眠": -2, "害怕": -2, "心惊胆战": -2, "心慌": -2,
    "提心吊胆": -2, "无助": -2, "气愤": -2, "流泪": -2,
    "烦躁": -2, "焦虑": -2, "生气": -2, "胆怯": -2,
    "难受": -2, "难过": -2, "震惊": -2,
    # Weight 1 (弱)
    "不安": -1, "不满": -1, "反感": -1, "吃惊": -1,
    "失落": -1, "忐忑": -1, "担心": -1, "烦": -1,
    "紧张": -1, "纠结": -1, "讨厌": -1, "迷茫": -1,
    "遗憾": -1, "郁闷": -1,

    # --- 正面词 (33个) ---
    # Weight 3 (强)
    "如释重负": 3, "幸福": 3,
    # Weight 2 (中)
    "信任": 2, "安心": 2, "尊重": 2, "幸运": 2,
    "庆幸": 2, "康复": 2, "开心": 2, "快乐": 2,
    "惊喜": 2, "愉快": 2, "感恩": 2, "感谢": 2,
    "放心": 2, "欢喜": 2, "欣慰": 2, "满意": 2,
    "痊愈": 2, "释然": 2, "顺利": 2, "高兴": 2,
    # Weight 1 (弱)
    "专业": 1, "出院": 1, "加油": 1, "喜欢": 1,
    "坚强": 1, "希望": 1, "推荐": 1, "耐心": 1,
    "负责": 1, "赞": 1, "鼓励": 1,
}

EMOTION_CATEGORIES = {
    "五雷轰顶": "惊", "噩梦": "惧", "天塌": "惧", "崩溃": "哀",
    "恐惧": "惧", "恐慌": "惧", "悲伤": "哀", "愤怒": "怒",
    "抑郁": "哀", "折磨": "哀", "晴天霹雳": "惊", "煎熬": "哀",
    "痛苦": "哀", "绝望": "哀",
    "伤心": "哀", "厌恶": "恶", "后悔": "哀", "哭": "哀",
    "失眠": "惧", "害怕": "惧", "心惊胆战": "惧", "心慌": "惧",
    "提心吊胆": "惧", "无助": "哀", "气愤": "怒", "流泪": "哀",
    "烦躁": "怒", "焦虑": "惧", "生气": "怒", "胆怯": "惧",
    "难受": "哀", "难过": "哀", "震惊": "惊",
    "不安": "惧", "不满": "怒", "反感": "恶", "吃惊": "惊",
    "失落": "哀", "忐忑": "惧", "担心": "惧", "烦": "怒",
    "紧张": "惧", "纠结": "惧", "讨厌": "恶", "迷茫": "惧",
    "遗憾": "哀", "郁闷": "怒",
    "如释重负": "乐", "幸福": "乐",
    "信任": "好", "安心": "乐", "尊重": "好", "幸运": "好",
    "庆幸": "好", "康复": "乐", "开心": "乐", "快乐": "乐",
    "惊喜": "惊", "愉快": "乐", "感恩": "好", "感谢": "好",
    "放心": "乐", "欢喜": "乐", "欣慰": "乐", "满意": "乐",
    "痊愈": "乐", "释然": "乐", "顺利": "乐", "高兴": "乐",
    "专业": "好", "出院": "乐", "加油": "好", "喜欢": "好",
    "坚强": "好", "希望": "好", "推荐": "好", "耐心": "好",
    "负责": "好", "赞": "好", "鼓励": "好",
}

NEGATION_WORDS = {"不", "没", "没有", "别", "莫", "勿", "未", "非", "无"}

# Sentiment scoring parameters (Methods, "Sentiment analysis")
# Lexicon terms are matched against the cleaned text as substrings, before
# segmentation, so that negation cues are preserved. A term is negated when any
# NEGATION_WORDS entry occurs in the NEGATION_WINDOW_CHARS characters immediately
# preceding it (four characters ~ two Chinese words).
NEGATION_WINDOW_CHARS = 4

# Length-adaptive net-score thresholds, calibrated on a 200-post development
# sample: (minimum character count, threshold). A text is Positive when the net
# score exceeds the threshold, Negative when it falls below its negative, and
# Neutral otherwise (including when no lexicon term is matched).
LENGTH_THRESHOLDS = ((501, 2), (100, 1), (0, 0))


def net_threshold(n_chars):
    """Return the net-score threshold for a text of n_chars characters."""
    for min_chars, thr in LENGTH_THRESHOLDS:
        if n_chars >= min_chars:
            return thr
    return 0
