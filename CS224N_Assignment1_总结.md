# CS224N Assignment 1 知识笔记：词向量探索

> **课程**: Stanford CS224N 2024 Spring · NLP with Deep Learning
> **作业**: Assignment 1 — Exploring Word Vectors (25分)
> **完成日期**: 2024年3月27日

---

## 目录

1. [核心问题：计算机如何"理解"词语？](#一核心问题)
2. [Part 1：从零构建计数型词向量](#二part-1从零构建计数型词向量)
   - [1.1 构建词表](#11-构建词表)
   - [1.2 构建共现矩阵](#12-构建共现矩阵)
   - [1.3 SVD降维](#13-svd降维)
   - [1.4 可视化词向量](#14-可视化词向量)
   - [1.5 聚类分析：共现矩阵能捕捉什么、不能捕捉什么](#15-聚类分析)
3. [Part 2：使用GloVe预测型词向量](#三part-2使用glove预测型词向量)
   - [2.1 GloVe vs 共现矩阵：可视化对比](#21-glove-vs-共现矩阵)
   - [2.2 余弦相似度：为什么不用欧氏距离？](#22-余弦相似度)
   - [2.3 多义词：一个向量装不下两个意思](#23-多义词)
   - [2.4 同义词与反义词：反义词为什么靠得近？](#24-同义词与反义词)
   - [2.5 词类比：向量算术的魔法](#25-词类比)
   - [2.6 类比失败：什么时候向量算术不灵？](#26-类比失败)
   - [2.7-2.9 词向量中的偏见](#27-29-词向量中的性别偏见)
4. [关键洞察总结](#四关键洞察总结)

---

## 一、核心问题

**计算机怎么"理解"一个词？** 给它一个向量。

但这个向量不能随便给。词向量（word vector / word embedding）的核心思想是：把每个词映射到高维空间中的一个点，让语义相近的词在空间中彼此靠近。而"语义相近"怎么定义？历史上两条路：

| 方法 | 代表 | 核心逻辑 |
|------|------|----------|
| **计数型** (count-based) | 共现矩阵 + SVD | 统计词与词在上下文中的共现次数 |
| **预测型** (prediction-based) | word2vec, GloVe | 训练模型预测上下文词，用中间层的权重作为词向量 |

本次作业让你**亲手走完两条路**，从原始语料开始，一步步构建词向量、降维、可视化、对比分析。

---

## 二、Part 1：从零构建计数型词向量（10分）

### 前置：IMDB数据集与预处理

```python
from datasets import load_dataset
imdb_dataset = load_dataset("stanfordnlp/imdb")

START_TOKEN = '<START>'
END_TOKEN = '<END>'
NUM_SAMPLES = 150

def read_corpus():
    files = imdb_dataset["train"]["text"][:NUM_SAMPLES]
    return [[START_TOKEN] + [re.sub(r'[^\w]', '', w.lower()) for w in f.split(" ")] + [END_TOKEN] for f in files]
```

**预处理细节**：
- 取训练集前150条影评
- 全部小写化
- 用正则 `[^\w]` 去掉所有标点符号（只保留字母数字下划线）
- 每条评论首尾分别加上 `<START>` 和 `<END>` token
- 返回的是一个两层嵌套列表：`[[word1, word2, ...], [word1, word2, ...], ...]`

**为什么要加 `<START>` 和 `<END>`？** 因为句首和句尾的词也需要上下文。`<START>` 作为"文档开始"的信号参与共现计数，让首词的左侧也有共现对象。

### 1.1 构建词表

> **题目（中文）**：实现 `distinct_words` 函数，找出语料中所有不重复的词，返回排序后的列表和数量。

```python
def distinct_words(corpus):
    corpus_words = []
    n_corpus_words = -1
    
    # 列表推导式展平嵌套列表
    all_words = [word for sentence in corpus for word in sentence]
    # 等价于：
    # all_words = []
    # for sentence in corpus:
    #     for word in sentence:
    #         all_words.append(word)
    
    # 去重 + 排序
    corpus_words = sorted(list(set(all_words)))
    n_corpus_words = len(corpus_words)
    
    return corpus_words, n_corpus_words
```

**关键操作链**：嵌套列表展平 → `set()` 去重 → `sorted()` 排序

**这一步在整个流程中的位置**：词表是所有后续操作的**索引基础**。共现矩阵的行和列都对应词表中的词，`word2ind` 字典就是从这里衍生出来的。

**Sanity check 通过**：用 toy corpus（"All that glitters isn't gold" / "All's well that ends well"）验证，输出 `Passed All Tests!`

---

### 1.2 构建共现矩阵

> **题目（中文）**：实现 `compute_co_occurrence_matrix` 函数，对给定语料和窗口大小（默认4），构建 word-by-word 共现矩阵。

**共现矩阵是什么？**

以窗口大小 n=1 为例，给定两句话：
- Doc1: `<START> all that glitters is not gold <END>`
- Doc2: `<START> all is well that ends well <END>`

构建出的矩阵中，每个格子 (i,j) 表示词 j 在词 i 的窗口内出现了多少次。例如 `all` 行、`<START>` 列的值是 2，因为两个文档中 `<START>` 都在 `all` 的窗口内。

```python
def compute_co_occurrence_matrix(corpus, window_size=4):
    words, n_words = distinct_words(corpus)
    M = np.zeros((n_words, n_words))  # 全零矩阵
    word2ind = {}
    
    for i, word in enumerate(words):
        word2ind[word] = i
    
    for sentence in corpus:
        for i, center_word in enumerate(sentence):
            center_index = word2ind[center_word]
            
            # 窗口边界
            left = max(0, i - window_size)
            right = min(len(sentence), i + window_size + 1)
            
            for j in range(left, right):
                if j == i:
                    continue  # 跳过中心词自己
                context_word = sentence[j]
                context_index = word2ind[context_word]
                M[center_index][context_index] += 1
    
    return M, word2ind
```

**核心逻辑**：
1. 遍历每个词作为**中心词**（center word）
2. 检测其前后 `window_size` 范围内的所有词作为**上下文词**（context word）
3. 在矩阵对应的 `[center, context]` 位置 +1
4. 跳过中心词自身（不让自己和自己共现）

**关键性质**：
- 矩阵是 `n_words × n_words` 大小
- **对称的**：词A出现在词B窗口的次数 ≈ 词B出现在词A窗口的次数
- **高度稀疏**：大部分词对从未在窗口内同时出现过

**Sanity check 通过**：用 toy corpus（window_size=1）和正确答案矩阵比对，输出 `Passed All Tests!`

**这一步在整个流程中的位置**：共现矩阵是 Part 1 的核心产出物。它是一条条"高维向量"——矩阵的每一行就是一个词的向量表示，维度等于词表大小（150条影评约产生 7902 个不同词，所以每个向量是 7902 维）。

---

### 1.3 SVD降维

> **题目（中文）**：实现 `reduce_to_k_dim` 函数，用 TruncatedSVD 把共现矩阵降到 k 维。

```python
def reduce_to_k_dim(M, k=2):
    n_iters = 10
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))
    
    svd = TruncatedSVD(n_components=k, n_iter=n_iters)
    M_reduced = svd.fit_transform(M)
    
    print("Done.")
    return M_reduced
```

**为什么要降维？**

原始共现矩阵是 7902×7902 的，有三大问题：
1. **维度灾难**：7902 维对于后续计算来说太大了
2. **稀疏性**：大部分值是零，信息密度低
3. **噪声**：偶然共现带来的随机波动

**SVD做了什么**：保留数据中方差最大的 k 个方向（主成分），把 7902 维的向量压缩到 k 维。语义相近的词在这些主方向上的投影也会接近。

**注意**：`TruncatedSVD.fit_transform()` 返回的是 `U × S`（左奇异向量乘以奇异值），不是单纯的 `U`。所以降维后还需要做 **L2归一化**，让所有向量落在单位圆上：

```python
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis]
```

**为什么要归一化？** 因为 SVD 返回的是 `U×S`，不同词的向量模长可能不同。归一化后，两个词的"接近程度"完全由方向决定，不受模长干扰。这本质上就是余弦相似度。

---

### 1.4 可视化词向量

> **题目（中文）**：实现 `plot_embeddings` 函数，把降维后的词向量画在二维平面上。

```python
def plot_embeddings(M_reduced, word2ind, words):
    for word in words:
        idx = word2ind[word]
        x, y = M_reduced[idx]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y, word)
    plt.show()
```

**工作原理**：
- 对要展示的每个词，从降维矩阵中取出它的 (x, y) 坐标
- 用红色 `x` 标记位置
- 在标记旁标注单词文本

**Sanity check**：用5个测试点的十字形排列验证，生成图应与 `question_1.4_test.png` 一致。

---

### 1.5 聚类分析

> **题目（中文）**：对 IMDB 影评语料运行完整 pipeline（window_size=4, SVD降维到2维，L2归一化），分析以下10个词的聚类情况：
> `['movie', 'book', 'mysterious', 'story', 'fascinating', 'good', 'interesting', 'large', 'massive', 'huge']`

运行这段代码：
```python
imdb_corpus = read_corpus()
M_co_occurrence, word2ind_co_occurrence = compute_co_occurrence_matrix(imdb_corpus, window_size=4)
M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis]

words = ['movie', 'book', 'mysterious', 'story', 'fascinating', 'good', 'interesting', 'large', 'massive', 'huge']
plot_embeddings(M_normalized, word2ind_co_occurrence, words)
```

输出：`Running Truncated SVD over 7902 words... Done.`

**回答问题 (a)**：找出至少两组聚类在一起的词，解释原因。

> **答案**：有两组明显的聚类：
> 1. **`book` 和 `movie`**：两者都是叙事媒介，在影评语料中频繁与"情节"、"角色"、"推荐"等相似上下文词共现。
> 2. **`good`、`interesting`、`fascinating`**：三者都是正面评价词，在影评中往往出现在相似的位置（"这部电影很___"）。

**回答问题 (b)**：你认为应该聚类但没有聚在一起的，举至少两个例子。

> **答案**：
> 1. **`large` 和 `huge`**：两者是近义词，但在这张图上分开了。可能因为它们在影评语料中搭配模式不同——"large"可能更常与"large amount"搭配，而"huge"可能与"huge disappointment"搭配，导致它们的上下文分布有差异。
> 2. **`book` 和 `story`**：理论上"book"和"story"的关系应该比"book"和"movie"更紧密，但实际情况相反。因为 IMDB 是**电影**评论数据集，"movie"和"story"的上下文远比"book"丰富。

**核心洞察**：共现矩阵的方法能捕捉到一部分语义关系，但由于它只是做简单的**词频计数**，对一些细微的语义差异不敏感。`large` 和 `huge` 在字典里是近义词，但在160条影评的具体语境中出现了不同的搭配习惯，纯计数方法就被误导了。

---

## 三、Part 2：使用GloVe预测型词向量（15分）

### 模型加载

```python
import gensim.downloader as api
wv_from_bin = api.load("glove-wiki-gigaword-200")
print("Loaded vocab size %i" % len(list(wv_from_bin.index_to_key)))
```

输出：`Loaded vocab size 400000`

**GloVe (Global Vectors) 是什么？**

GloVe 是一种"混合方法"：它既统计全局词共现（像 Part 1 那样），又用神经网络优化一个目标函数来学习词向量。训练语料是 Wikipedia + Gigaword，产出 40 万个词的 200 维稠密向量。

**和 Part 1 共现矩阵的关键区别**：
- 共现矩阵：维度 = 词表大小（7902），稀疏，信息利用粗糙
- GloVe：固定维度（200），稠密，每个维度都编码了丰富的语义信息

---

### 2.1 GloVe vs 共现矩阵：可视化对比

> **题目（中文）**：用同样的10个词，将 GloVe 的 200维向量降到2维并可视化，与 Part 1.5 的共现矩阵图对比。

```python
M, word2ind = get_matrix_of_vectors(wv_from_bin, words)
M_reduced = reduce_to_k_dim(M, k=2)
M_lengths = np.linalg.norm(M_reduced, axis=1)
M_reduced_normalized = M_reduced / M_lengths[:, np.newaxis]

plot_embeddings(M_reduced_normalized, word2ind, words)
```

**观察**：GloVe 的聚类效果明显优于共现矩阵——同义/同类的词在 GloVe 空间中更紧凑。

**但这不重要。真正重要的是**：从 200 维降到 2 维会损失多少信息？即便如此，语义关系依然得到了保留。这说明 200 维的 GloVe 空间中存在一个**低维语义流形**，词之间的关系不是随机散乱的，而是有结构的。

---

### 2.2 余弦相似度

> **题目（中文）**：对 `monstrous`、`loving`、`limpid` 等词，用余弦相似度找最近邻。解释为什么 NLP 中用余弦相似度而不是欧几里得距离。

**运行示例（notebook中实际输出）**：

以 `monstrous` 为例：
```
[('monstrously', 0.6904), ('hideous', 0.6628), ('horrendous', 0.6564),
 ('monstrosity', 0.6509), ('grotesque', 0.6448), ...]
```

以 `loving` 为例：
```
[('lovingly', 0.7297), ('lovable', 0.6727), ('caring', 0.6725),
 ('affectionate', 0.6687), ('devoted', 0.6600), ...]
```

**为什么用余弦相似度而不是欧氏距离？**

词向量的**方向**比**模长**更有意义。高频词（the, a, is）的向量模长通常很大——它们无处不在，和谁都有共现。如果按欧氏距离，the 会和所有词都很"接近"，这没有意义。

余弦相似度 = 两个向量夹角的余弦值 = **归一化后的点积**。它只关心"指向哪个方向"，不考虑"有多长"。所以余弦相似度对高频词天然免疫。

**类比**：两个箭矢，一个 10 米长指向东，一个 1 米长也指向东。欧氏距离说它们差 9 米；余弦相似度说它们完全一致。在 NLP 中，"指向东" = "语义模式一致"，箭矢长度不重要。

---

### 2.3 多义词

> **题目（中文）**：找两个有多重含义的词。解释用一个固定向量表示多义词有什么局限性。

> **答案**：像 "bank" 这种词，有"银行"和"河岸"两个完全不同的含义。GloVe 只能给 "bank" **一个** 200 维向量。这个向量是对所有出现 "bank" 的上下文取**加权平均**后的结果，所以它既不完全是"银行"，也不完全是"河岸"，而是两者语义的模糊混合物。

**这是一个深刻的局限**：词向量是**静态的**——一个词不管出现在什么上下文中，向量都一样。但现实中词语的含义高度依赖于上下文。

> "I went to the bank to withdraw money."（银行）
> "I sat on the bank of the river."（河岸）

同一个 "bank"，完全不同的意思。GloVe 用同一个向量来表示它们，必然会丢失信息。

**这引出了什么？** 后来的 **BERT / Transformer** 等模型正是为了解决这个问题而设计的——它们生成的是**上下文相关**（contextualized）的词表示，同一个词在不同句子中得到不同的向量。

---

### 2.4 同义词与反义词

> **题目（中文）**：各找一组同义词和反义词，用余弦相似度测量它们的接近程度。解释为什么反义词有时在向量空间中很接近。

**同义词示例**：`happy` 和 `glad`
```
happy → ('glad', 0.7409), ('pleased', 0.6692), ('delighted', 0.6252), ...
```

**反义词示例**：`hot` 和 `cold`
```
hot → ('cold', 0.6042), ('warm', 0.5986), ('heat', 0.5819), ...
```

**反直觉的现象**：`hot` 和 `cold` 的余弦相似度高达 **0.60**，而随机无关词的相似度通常在 0.1-0.3。反义词竟然比随机词"更接近"。

**原因**：词向量捕捉的是**上下文分布的相似性**，不是**语义等价性**。

"hot" 和 "cold" 虽然意思相反，但它们的**使用场景高度重叠**：
- 都描述温度：The water is hot / cold.
- 都描述食物：The soup is hot / cold.
- 都描述天气：It's hot / cold outside.
- 在句子中占据相同的语法位置（谓语、定语）

所以"hot"和"cold"的上下文分布几乎一样，它们的词向量自然就很接近。

**这是整个作业最重要的洞察之一**：

> **词向量学到的 = 词在语料中的分布模式 ≈ 词常和哪些词一起出现**

它不是一个"语义词典"。它是训练语料中**共现统计的压缩编码**。

---

### 2.5 词类比：向量算术

> **题目（中文）**：找到一组词 (x, y, a, b)，使得 x:y :: a:b。即 `vec(a) + vec(y) - vec(x)` 最接近 `vec(b)`。验证 king:man :: queen:woman。

```python
x, y, a, b = ("king", "queen", "man", "woman")

# 验证
assert wv_from_bin.most_similar(positive=[a, y], negative=[x])[0][0] == b
# → 通过。most_similar(positive=['man', 'queen'], negative=['king']) 返回 'woman'
```

**这是什么魔法？**

$$ \vec{queen} \approx \vec{king} - \vec{man} + \vec{woman} $$

用向量运算表达了一个**关系类比**："king 对 man 的关系 ≈ queen 对 woman 的关系"。更直白地说：

$$ \vec{king} - \vec{man} \approx \vec{queen} - \vec{woman} $$

两者的 "差值向量" 都指向"王室身份"的方向。

**为什么可以这样？** 因为 GloVe 训练出的向量空间中，**语义关系被线性编码**了。"性别方向"（male↔female）、"皇室方向"（commoner↔royalty）在向量空间中都有各自的近似方向。虽然模型没有显式学习"皇室关系"，但训练语料中 "king" 和 "man" 的上下文差异模式，恰好和 "queen" 与 "woman" 的上下文差异模式相似——这些统计规律自然形成了向量空间的这种结构。

---

### 2.6 类比失败

> **题目（中文）(a)**：`hand : glove :: foot : ?` 预期返回 `sock`，实际返回了什么？为什么？

```python
pprint.pprint(wv_from_bin.most_similar(positive=['foot', 'glove'], negative=['hand']))
```

**实际输出**：
```
[('45,000-square', 0.4922),
 ('15,000-square', 0.4649),
 ('10,000-square', 0.4544),
 ('6,000-square', 0.4497),
 ('3,500-square', 0.4441),
 ('700-square', 0.4425),
 ('50,000-square', 0.4356),
 ('3,000-square', 0.4348),
 ('30,000-square', 0.4330),
 ('footed', 0.4323)]
```

全是 `XX-square` 的面积单位，完全没有 `sock`。

**原因分析**：
1. "glove"（手套）在训练语料（Wikipedia + Gigaword）中更常出现在**体育装备**的上下文中：棒球手套 (baseball glove)、拳击手套 (boxing glove)，而非日常衣物。所以 glove 的向量偏向"运动装备"。
2. "foot" 和 "hand" 在语料中大多只是身体部位，它们之间的"功能对应关系"（手套:手 :: 袜子:脚）在训练数据中的统计信号太弱。
3. 词向量能学到的是**统计上频繁的共同出现模式**（共现相似性），而不是**概念层面的功能对应**（功能类比）。

> **题目 (b)**：自己找一个不成立的类比。

```python
x, y, a, b = ("tokyo", "japan", "paris", "france")
pprint.pprint(wv_from_bin.most_similar(positive=[a, y], negative=[x]))
# → 返回结果中首位的不是 france，assert b != 通过
```

**输出**（前10个）：
```
[('france', 0.8588), ('french', 0.6991), ('europe', 0.6315),
 ('prohertrib', 0.6207), ('belgium', 0.5940), ('britain', 0.5821),
 ('italy', 0.5818), ('european', 0.5652), ('germany', 0.5589), ('spain', 0.5508)]
```

注意，`france` 确实是第一个结果（0.8588 的余弦相似度最高），但紧随其后的是一串欧洲国家名，说明**地理实体在语义空间中高度聚类**——"城市→国家"的关系并不总是一个清晰的线性方向。

**核心洞察**：要理解为什么某些类比成立、某些失败，必须回到**训练语料**中去找答案。词向量不是万能的，它只是语料中统计模式的忠实记录。如果一个关系模式在语料中不够显著，模型就"学不到"。

---

### 2.7-2.9 词向量中的性别偏见

> **题目 2.7（中文）**：运行预置代码，观察 `man + profession - woman` 和 `woman + profession - man` 的相似词差异。

```python
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'profession'], negative=['woman']))
print()
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'profession'], negative=['man']))
```

**实际输出对比**：

| man + profession - woman | woman + profession - man |
|--------------------------|--------------------------|
| reputation (0.525) | professions (0.596) |
| professions (0.518) | practitioner (0.499) |
| skill (0.490) | teaching (0.483) |
| skills (0.490) | nursing (0.482) |
| ethic (0.490) | vocation (0.479) |
| business (0.488) | teacher (0.472) |
| respected (0.486) | practicing (0.469) |
| practice (0.482) | educator (0.465) |
| regarded (0.478) | physicians (0.463) |
| life (0.476) | professionals (0.460) |

**一目了然**：左侧偏向"声誉、技能、商业、受人尊敬"，右侧偏向"教学、护理、教育者"。

---

> **题目 2.8（中文）**：自己设计另一个类比，发现词向量中的偏见。

```python
pprint.pprint(wv_from_bin.most_similar(positive=['man', 'doctor'], negative=['woman']))
print("-----------------------------------------")
pprint.pprint(wv_from_bin.most_similar(positive=['woman', 'nurse'], negative=['man']))
```

**实际输出**：

| man + doctor - woman | woman + nurse - man |
|----------------------|---------------------|
| dr. (0.549) | nurses (0.644) |
| physician (0.533) | pregnant (0.611) |
| he (0.528) | midwife (0.591) |
| him (0.523) | mother (0.563) |
| himself (0.512) | nursing (0.563) |
| medical (0.505) | therapist (0.555) |
| his (0.504) | anesthetists (0.543) |
| brother (0.503) | anesthetist (0.535) |
| surgeon (0.501) | pediatrician (0.525) |
| mr. (0.494) | dentist (0.519) |

**发现**：`man + doctor - woman` 的关联词包含大量**男性代词**（he, him, himself, his），而 `woman + nurse - man` 的关联词包含 `pregnant`、`midwife`、`mother`。"医生=男性、护士=女性"的刻板印象被精确地编码在了向量空间中。

---

> **题目 2.9(a)（中文）**：偏见是怎么进入词向量的？给一个真实例子。

> **答案**：词向量中的偏见来源于训练语料中的统计共现模式。如果在大量真实文本中，"nurse" 更多与女性代词（she, her）共现，而 "doctor" 更多与男性代词（he, his）共现，模型就会忠实地学习并编码这种统计关联。这不是模型"想要"歧视，而是**训练数据本身就反映了历史上和现实中的性别不平等**。模型只是一个诚实的镜子。

> **题目 2.9(b)（中文）**：提出一种减轻词向量偏见的方法。

> **答案**：一种常见的方法是**去偏处理（debiasing）**：
> 1. 识别向量空间中的"性别方向"：计算 `he - she` 或 `man - woman` 的向量差，近似作为性别子空间的主轴。
> 2. 对需要中和的词（如 profession 类词汇），将其投影到与性别方向正交的子空间，消除其性别维度上的分量。
> 3. 例如，去偏后 "doctor" 和 "nurse" 将不再强烈依赖性别维度，它们的位置更多由专业技能相关的语义维度决定。

---

## 四、关键洞察总结

### 4.1 两条路线的本质区别

| | 计数型（Part 1） | 预测型（Part 2 / GloVe） |
|---|---|---|
| **输入** | 原始语料的共现计数 | 预训练的稠密向量 |
| **维度** | 词表大小（7902），稀疏 | 固定（200），稠密 |
| **语义捕捉** | 粗略，噪声大 | 精细，捕捉到类比和细微关系 |
| **工程成本** | 轻量，几分钟 | 需要下载预训练模型（252MB） |
| **可解释性** | 高（每个维度是可追溯的共现次数） | 低（每个维度是不可解释的隐变量） |

### 4.2 最重要的三个概念洞察

**洞察1：词向量捕捉的是上下文分布，不是语义本身。**

这是贯穿整个作业的核心结论。证据链：
- 2.4：反义词 hot/cold 相似度高，因为它们出现在完全相同的上下文中
- 2.6：hand:glove 类比失败，因为 glove 的共现环境偏向运动装备而非日常衣物
- 2.7-2.8：职业偏见来自训练语料中性别与职业的统计共现模式

词向量不是一本"语义字典"，它是**语料中词汇共现统计的压缩编码**。这个区别决定了它的能力和边界。

**洞察2：向量空间中的线性运算可以捕捉语义关系。**

king - man + woman ≈ queen。这不是刻意设计的，而是**涌现属性**——语义关系的线性结构是训练数据中这些词共现模式的自然结果。这个发现（Mikolov et al., 2013）改变了 NLP 的发展方向。

**洞察3：每个词一个向量是根本性的局限。**

多义词（2.3）和偏见（2.7-2.9）都源于同一个问题：静态词向量对所有上下文一视同仁。后来的上下文相关模型（ELMo, BERT, GPT）正是在解决这个问题——根据上下文动态生成词表示。

### 4.3 这份作业教了什么？

1. **从原理出发构建**：不是调一句 `Word2Vec.load()` 就完事，而是要理解"共现计数 → 高维矩阵 → SVD降维 → 归一化"的完整链路。
2. **用数据说话**：所有的分析都基于实际运行结果和可视化，不是空对空的理论。
3. **批判性地看待模型输出**：既看到词向量的强大（类比、语义聚类），也看到它的不足（多义词、偏见）。
4. **技术与社会不可分**：从统计共现到性别偏见只有一步之遥，技术选择有社会后果。

---

### 附录：技术栈速查

| 库 | 关键函数 | 用途 |
|----|----------|------|
| `numpy` | `np.zeros()`, `np.linalg.norm()` | 矩阵创建、L2归一化 |
| `sklearn.decomposition.TruncatedSVD` | `fit_transform()` | 截断SVD降维 |
| `matplotlib.pyplot` | `scatter()`, `annotate()` | 词向量可视化 |
| `gensim.downloader` | `api.load("glove-wiki-gigaword-200")` | 加载预训练GloVe |
| `gensim.models.KeyedVectors` | `most_similar(positive, negative)` | 余弦相似度查询、词类比 |
| `datasets` (HuggingFace) | `load_dataset("stanfordnlp/imdb")` | 加载IMDB影评 |
| `re` | `re.sub(r'[^\w]', '', w.lower())` | 文本清洗 |
