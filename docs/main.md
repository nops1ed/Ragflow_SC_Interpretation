# Rag FLow Source Code Interpretation
by Yingze Sun

## Index
- 结构

## Why Rag Flow ? What's new ? How to implement ?
在聊这个问题之前，先谈谈RAG的一些基础知识，熟悉这些的同志可跳过这段内容。

### 大模型知识库的构建
大模型爆火之后，部分企业发现现有的大模型无法回答一些公司个性化的内容，因此就出现了针对大模型的知识库，通过提供公司内部的背景知识提升大模型个性化回答的能力，但这背后涉及到信息泄露，因此大部分公司会采用私有化部署的模型来解决该问题，这就是私有化部署的企业知识库。

### Tech Solution
目前大模型知识库的构建一般是使用检索增强生成 (RAG) 的技术方案，RAG 首先会检索与问题相关的文档，之后会将检索到的信息提供给大模型（LLM），LLM 使用它们来生成更准确、更相关的响应。RAG 有三个核心组件：
- 检索组件（Retriever）
    - 文档库
        - 文档存储：存储大量的文档或知识片段，通常是一个大型的文本数据库或知识库。
        - 索引结构：使用高效的索引结构（如倒排索引、向量索引）来加速检索过程。
    - 检索算法
        - BM25：一种基于词频和逆文档频率的传统检索算法。
        - 向量检索：使用向量表示（如BERT嵌入）和相似度度量（如余弦相似度）进行检索。
        - 混合检索：结合传统检索算法和向量检索算法，提升检索效果。
- 生成组件（Generator）
    - 预训练生成模型
    - 生成策略
        - 贪婪搜索：每一步选择概率最高的词，生成文本。
        - 束搜索：在每一步保留多个候选词，选择最优路径生成文本。
        - 采样方法：如Top-k采样和Top-p采样，增加生成文本的多样性。
- 融合组件（Fusion）
    - 融合策略
        - 直接拼接：将检索到的文档直接拼接到输入中，作为生成模型的上下文。
        - 加权融合：根据检索结果的相关性分配权重，融合到生成模型的输入中。
        - 注意力机制：使用注意力机制动态选择和融合检索到的文档

WorkFlow:
```c
Local      ---> Unstructed --> Text ---> Text Splitter ---> Text Chunks
Documents         Loader                                        |
                                                                |
                                                                v
                                                            Embedding
                                                                |
                                                                |
                                                                v
Query      --->     Embedding --->  Query Vector --->  Vector Similarity <--- VectorStore
                                            |
                                            |
                                            v
            Prompt  <---  Prompt <--- Related Text chunks
              |          Template
              |
              v
Answer <---  LLM

                                                            //Langchain + ChatGLM
```
TODO：

### 方案落地
NULL

### 评估体系
参考[赛博恩师吴恩达的视频](https//www.bilibili.com/video/BV1494y1E7H9)


-----
让我们回到Ragflow, 回答最初始的问题: Why, What, How

### Why RagFlow ? What is RagFlow; Key features ? 
- Quality in, quality out; 细粒度文档解析，[官方文档](https://github.com/infiniflow/ragflow?tab=readme-ov-file#-key-features)如此描述：
    - Deep document understanding-based knowledge extraction from unstructured data with complicated formats.
    - Finds "needle in a data haystack" of literally unlimited tokens.

- ...

### How ? ---> Rag Flow 工作流
![](./res/Ragflow_arch.png)
对比之前的Langchain可以看出，右侧的知识库更加庞大，加入了更多文档解析的功能：OCR, Documnet Layout Analyze.
实际上,RagFlow没有使用任何 RAG 中间件，而是完全重新研发了一套智能文档理解系统，并以此为依托构建 RAG 任务编排体系，也可以理解文档的解析是其 RagFlow 的核心亮点。

### 代码结构
```c
// Current version
- agent/
- api/
- conf/
- deepdoc/
- graphrag/
- rag/
-----------------
- docs/
```
现版本的文件结构：
- Web 服务
- 业务数据库使用的是 MySQL
- 向量数据库使用的是 ElasticSearch ，奇怪的是公司有自己的向量数据库 infinity 竟然默认没有用上
- 文件存储使用的是 MinIO
- GraphRAG : 启发于 graphrag 和思维导图

正如前面介绍的因为没有使用 RAG 中间件，比如 langchain 或 llamaIndex，因此实现上与常规的 RAG 系统会存在一些差异

### 文件加载
让我们从RagFlow的使用开始抽丝剥茧，逐渐深度到代码中；首先就是和用户深度交互的文档解析界面

常规的 RAG 服务都是在上传时进行文件的加载和解析，但是 RAGFlow 的上传仅仅包含上传至 MinIO，需要手工点击触发文件的解析。
![](./res/sample.png)
实际上，仅是几KB的文档，解析速度也令人堪忧，考虑到资源开销比较大，因此也能理解为什么采取二次手工确认的产品方案了。
对应的文档处理函数位于*RAGFLOW_HOME/api/db/services/task_services.py*

```c
def queue_tasks(doc, bucket, name):
    def new_task():
        nonlocal doc
        return {
            "id": get_uuid(),
            "doc_id": doc["id"]
        }
    tsks = []

    if doc["type"] == FileType.PDF.value:
        file_bin = MINIO.get(bucket, name)
        do_layout = doc["parser_config"].get("layout_recognize", True)
        pages = PdfParser.total_page_number(doc["name"], file_bin)
        page_size = doc["parser_config"].get("task_page_size", 12)
        if doc["parser_id"] == "paper":
            page_size = doc["parser_config"].get("task_page_size", 22)
        if doc["parser_id"] == "one":
            page_size = 1000000000
        if doc["parser_id"] == "knowledge_graph":
            page_size = 1000000000
        if not do_layout:
            page_size = 1000000000
        page_ranges = doc["parser_config"].get("pages")
        if not page_ranges:
            page_ranges = [(1, 100000)]
        for s, e in page_ranges:
            s -= 1
            s = max(0, s)
            e = min(e - 1, pages)
            for p in range(s, e, page_size):
                task = new_task()
                task["from_page"] = p
                task["to_page"] = min(p + page_size, e)
                tsks.append(task)

    elif doc["parser_id"] == "table":
        file_bin = MINIO.get(bucket, name)
        rn = RAGFlowExcelParser.row_number(
            doc["name"], file_bin)
        for i in range(0, rn, 3000):
            task = new_task()
            task["from_page"] = i
            task["to_page"] = min(i + 3000, rn)
            tsks.append(task)
    else:
        tsks.append(new_task())

    bulk_insert_into_db(Task, tsks, True)
    DocumentService.begin2parse(doc["id"])

    for t in tsks:
        assert REDIS_CONN.queue_product(SVR_QUEUE_NAME, message=t), "Can't access Redis. Please check the Redis' status."
```

可以看出，该函数将文档拆分成一个或多个任务异步处理，具体地，**queue_task()**会对文档进行识别分类，根据不同的类型设置单个任务最多处理的页数，默认分为12页，若是paper类型的pdf，则分22页，其余不分页，并通过 Redis 消息队列进行暂存，之后就可以离线异步处理


Redis消息队列的消费模块:
```c
// 省略的异常处理和一些其他信息，保留了代码骨架，方便阅读
def main():
    rows = collect()
    for _, r in rows.iterrows():
        embd_mdl = LLMBundle(r["tenant_id"], LLMType.EMBEDDING, llm_name=r["embd_id"], lang=r["language"])
        if r.get("task_type", "") == "raptor":
            chat_mdl = LLMBundle(r["tenant_id"], LLMType.CHAT, llm_name=r["llm_id"], lang=r["language"])
                cks, tk_count = run_raptor(r, chat_mdl, embd_mdl, callback)
        else:
            st = timer()
            cks = build(r)
            tk_count = embedding(cks, embd_mdl, r["parser_config"], callback)

        init_kb(r)
        chunk_count = len(set([c["_id"] for c in cks]))
        es_r = ELASTICSEARCH.bulk(cks[b:b + es_bulk_size], search.index_name(r["tenant_id"]))

```
整体处理流程如下：
1. 调用 collect() 方法从Redis消息队列中获取任务
2. 每个任务会依次调用 build() 进行文件的解析
3. 调用 embedding() 方法进行向量化
4. 最后调用 ELASTICSEARCH.bulk() 写入 ElasticSearch

之前谈到文档的解析是其 RagFlow 的核心亮点，build()方法处理文档并将其分块。它从 MINIO 存储中获取文档的二进制数据，使用指定的解析器将文档分块，并将结果存储在 MINIO 中。下面看一下build()方法的具体逻辑：

```c
// 省略的异常处理和一些其他信息，保留了代码骨架，方便阅读
def build(row):
    chunker = FACTORY[row["parser_id"].lower()]
    bucket, name = File2DocumentService.get_minio_address(doc_id=row["doc_id"])
    binary = get_minio_binary(bucket, name)
    cks = chunker.chunk(row["name"], binary=binary, from_page=row["from_page"],
                        to_page=row["to_page"], lang=row["language"], callback=callback,
                        kb_id=row["kb_id"], parser_config=row["parser_config"], tenant_id=row["tenant_id"])
```
这里谈到的解析器以**parser_id**指定，追溯到原本的解析器组：
```c
FACTORY = {
    "general": naive,
    ParserType.NAIVE.value: naive,
    ParserType.PAPER.value: paper,
    ParserType.BOOK.value: book,
    ParserType.PRESENTATION.value: presentation,
    ParserType.MANUAL.value: manual,
    ParserType.LAWS.value: laws,
    ParserType.QA.value: qa,
    ParserType.TABLE.value: table,
    ParserType.RESUME.value: resume,
    ParserType.PICTURE.value: picture,
    ParserType.ONE.value: one,
    ParserType.AUDIO.value: audio,
    ParserType.EMAIL.value: email,
    ParserType.KG.value: knowledge_graph
}
```
其实现位于**RAGFLOW_HOME/rag/app**下，以naive为例，其支持**docx, pdf, excel, txt**等主流类型的文档解析，我们以**pdf**格式为例：
```c
class Pdf(PdfParser):
```
该解析器继承自**RAGFLOW_HOME/deepdoc/parser/pdf_parser.py**中的*RAGFlowPdfParser*，pdf打开使用接口[pypdf](https://pypi.org/project/pypdf/#description)



