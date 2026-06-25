"""
水果识别 - 水果类别列表 (共23种)
"""

FRUITS = [
    # (英文目录名, 中文名, 百度搜索关键词)
    # ⚠️ 必须按字母序排列，与 ImageFolder.classes 的排序一致！
    ('apple',        '苹果',   ['苹果 水果', '红苹果 水果', 'apple fruit']),
    ('avocado',      '牛油果', ['牛油果 水果', 'avocado fruit']),
    ('banana',       '香蕉',   ['香蕉 水果', 'banana fruit']),
    ('blueberry',    '蓝莓',   ['蓝莓 水果', 'blueberry fruit']),
    ('cherry',       '樱桃',   ['樱桃 水果', '车厘子', 'cherry fruit']),
    ('coconut',      '椰子',   ['椰子 水果', 'coconut fruit']),
    ('dragon_fruit', '火龙果', ['火龙果 水果', '红心火龙果', 'dragon fruit']),
    ('durian',       '榴莲',   ['榴莲 水果', 'durian fruit']),
    ('grape',        '葡萄',   ['葡萄 水果', '紫葡萄', 'grape fruit']),
    ('kiwi',         '猕猴桃', ['猕猴桃 水果', 'kiwi fruit cut']),
    ('lemon',        '柠檬',   ['柠檬 水果', 'lemon fruit']),
    ('lychee',       '荔枝',   ['荔枝 水果', 'lychee fruit']),
    ('mango',        '芒果',   ['芒果 水果', 'mango fruit']),
    ('orange',       '橙子',   ['橙子 水果', 'orange fruit']),
    ('papaya',       '木瓜',   ['木瓜 水果', 'papaya fruit']),
    ('peach',        '桃子',   ['桃子 水果', '水蜜桃', 'peach fruit']),
    ('pear',         '梨',     ['梨 水果', '鸭梨', 'pear fruit']),
    ('pineapple',    '菠萝',   ['菠萝 水果', 'pineapple fruit']),
    ('pomegranate',  '石榴',   ['石榴 水果', 'pomegranate fruit']),
    ('star_fruit',   '杨桃',   ['杨桃 水果', 'star fruit carambola']),
    ('strawberry',   '草莓',   ['草莓 水果', 'strawberry fruit']),
    ('tomato',       '番茄',   ['番茄 水果', '西红柿', 'tomato fruit']),
    ('watermelon',   '西瓜',   ['西瓜 水果', '切开西瓜', 'watermelon fruit']),
]

NUM_CLASSES = len(FRUITS)

# 生成中英文映射
IDX_TO_CN = {i: f[1] for i, f in enumerate(FRUITS)}
IDX_TO_EN = {i: f[0] for i, f in enumerate(FRUITS)}
CN_TO_IDX = {f[1]: i for i, f in enumerate(FRUITS)}
CLASS_NAMES = [f[1] for f in FRUITS]  # GUI显示用中文
