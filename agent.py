import datetime
import decimal
import json
import os
import sys
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pymysql
from sqlglot import exp, parse


# ----------------------------
# 配置：优先读取同目录 .env，可用环境变量覆盖默认值
# ----------------------------

def load_dotenv(path: str = ".env") -> None:
    """加载 .env 配置；已存在的环境变量不会被覆盖。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv()

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "127.0.0.1"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "123456"),
    "database": os.getenv("MYSQL_DATABASE", "ods"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
    "connect_timeout": 5,
}

def _clean_env(value: str) -> str:
    """去掉配置值中可能混入的引号、反引号和空白。"""
    return value.strip().strip("`\"'").strip()


def _normalize_base_url(raw_url: str) -> str:
    """统一 base URL；兼容已带上 /chat/completions 的写法。"""
    url = _clean_env(raw_url).rstrip("/")
    if url.endswith("/chat/completions"):
        url = url[: -len("/chat/completions")]
    return url


LLM_BASE_URL = _normalize_base_url(
    os.getenv("OPENAI_BASE_URL")
    or os.getenv("LLM_BASE_URL")
    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
LLM_API_KEY = _clean_env(
    os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY") or ""
)
LLM_MODEL = _clean_env(
    os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "qwen-plus"
)

RELS_PATH = os.getenv("RELS_PATH", "rels.json")
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "200"))
MAX_REPAIR_ATTEMPTS = 1


# ----------------------------
# Agent State
# ----------------------------

@dataclass
class AgentState:
    question: str
    phase: str = "intent"
    intent: dict[str, Any] = field(default_factory=dict)
    catalog: list[dict[str, str]] = field(default_factory=list)
    selected_tables: list[str] = field(default_factory=list)
    schemas: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    relations: list[dict[str, str]] = field(default_factory=list)
    sql: str = ""
    validation_error: str = ""
    execution_error: str = ""
    result: list[dict[str, Any]] = field(default_factory=list)
    repair_attempts: int = 0
    done: bool = False


# ----------------------------
# 通用工具
# ----------------------------

def log(message: str) -> None:
    print(f"[Agent] {message}")


def quote_identifier(name: str) -> str:
    """对从数据库读取到的标识符加反引号。"""
    return "`" + name.replace("`", "``") + "`"


def call_llm(system_prompt: str, user_payload: dict[str, Any]) -> dict[str, Any]:
    """
    调用 OpenAI Chat Completions 兼容接口
    （默认阿里云百炼 DashScope，可通过 OPENAI_BASE_URL 切换其他兼容服务）。
    """
    if not LLM_API_KEY:
        raise RuntimeError("请先在 .env 或环境变量中设置 OPENAI_API_KEY。")

    body = {
        "model": LLM_MODEL,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    }

    request = urllib.request.Request(
        url=f"{LLM_BASE_URL}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"调用 LLM 失败：{exc}") from exc

    content = response_data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM 返回内容不是合法 JSON：{content}") from exc


# ----------------------------
# 数据库元数据与查询工具
# ----------------------------

def _to_json_value(value: Any) -> Any:
    """把 MySQL 返回的非 JSON 原生类型转为基本类型，保证结果可 json.dumps。"""
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.time, datetime.timedelta)):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class MySQLTools:
    def __init__(self, config: dict[str, Any]):
        self.config = config

    def connect(self):
        return pymysql.connect(**self.config)

    def get_catalog(self) -> list[dict[str, str]]:
        """只读取表名和表注释，供 LLM 初步选表。"""
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLE STATUS")
                rows = cursor.fetchall()

        return [
            {
                "table": row["Name"],
                "comment": row.get("Comment") or "",
            }
            for row in rows
        ]

    def get_table_schema(
        self, table_names: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """只加载已选表的字段和字段注释。"""
        schemas: dict[str, list[dict[str, Any]]] = {}

        with self.connect() as conn:
            with conn.cursor() as cursor:
                for table in table_names:
                    sql = f"SHOW FULL COLUMNS FROM {quote_identifier(table)}"
                    cursor.execute(sql)
                    rows = cursor.fetchall()
                    schemas[table] = [
                        {
                            "name": row["Field"],
                            "type": row["Type"],
                            "nullable": row["Null"] == "YES",
                            "key": row["Key"],
                            "comment": row.get("Comment") or "",
                        }
                        for row in rows
                    ]

        return schemas

    def execute_select(self, sql: str) -> list[dict[str, Any]]:
        """只执行校验通过的 SELECT，并限制返回行数；行值转为可 JSON 序列化的类型。"""
        with self.connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql)
                rows = cursor.fetchmany(MAX_RESULT_ROWS)

        return [
            {key: _to_json_value(value) for key, value in row.items()}
            for row in rows
        ]


def load_relations(path: str) -> list[dict[str, str]]:
    """
    支持当前 rels.json 的结构：

    {
      "表A": {
        "表B": [["表A字段", "表B字段"]]
      }
    }
    """
    with open(path, "r", encoding="utf-8") as file:
        raw = json.load(file)

    relations = []
    for left_table, targets in raw.items():
        if not isinstance(targets, dict):
            continue
        for right_table, pairs in targets.items():
            if not isinstance(pairs, list):
                continue
            for pair in pairs:
                if isinstance(pair, list) and len(pair) == 2:
                    relations.append({
                        "left_table": left_table,
                        "left_column": str(pair[0]),
                        "right_table": right_table,
                        "right_column": str(pair[1]),
                    })
    return relations


# ----------------------------
# SQL 安全校验
# ----------------------------

def validate_sql(
    sql: str,
    schemas: dict[str, list[dict[str, Any]]],
    relations: list[dict[str, str]],
) -> None:
    """拒绝非 SELECT、未知表/字段，以及不在 rels.json 中的 JOIN 条件。"""

    parsed = parse(sql, read="mysql")
    if len(parsed) != 1 or parsed[0] is None:
        raise ValueError("SQL 必须是单条语句。")

    tree = parsed[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("只允许 SELECT 查询。")

    table_names = {name.lower(): name for name in schemas}
    alias_to_table: dict[str, str] = {}
    used_tables: set[str] = set()

    for table_node in tree.find_all(exp.Table):
        # 不允许跨库引用。
        if table_node.db or table_node.catalog:
            raise ValueError("不允许引用其他数据库或 Catalog。")

        table_name = table_node.name
        canonical = table_names.get(table_name.lower())
        if canonical is None:
            raise ValueError(f"SQL 使用了未选中或不存在的表：{table_name}")

        used_tables.add(canonical)
        alias_to_table[table_name.lower()] = canonical

        alias = table_node.alias
        if alias:
            alias_to_table[alias.lower()] = canonical

    if not used_tables:
        raise ValueError("SQL 中没有可识别的表。")

    column_sets = {
        table: {str(c["name"]).lower() for c in columns}
        for table, columns in schemas.items()
    }

    def enclosing_select(node):
        """向上找到最近的 SELECT；UNION 的每个分支各自是一个独立作用域。"""
        node = node.parent
        while node is not None:
            if isinstance(node, exp.Select):
                return node
            node = node.parent
        return None

    def select_scope(select):
        """该 SELECT 作用域内可见的表，不包含其内部子查询的表。"""
        if select is None:
            return used_tables
        scope = set()
        for table_node in select.find_all(exp.Table):
            if enclosing_select(table_node) is select:
                scope.add(
                    alias_to_table.get(table_node.name.lower(), table_node.name)
                )
        return scope

    def resolve_column(column: exp.Column) -> tuple[str, str]:
        column_name = column.name
        qualifier = column.table

        if qualifier:
            table = alias_to_table.get(qualifier.lower())
            if table is None:
                raise ValueError(f"字段使用了未知表别名：{qualifier}")
            if column_name.lower() not in column_sets[table]:
                raise ValueError(f"表 {table} 不存在字段：{column_name}")
            return table, column_name
        else:
            # 只在当前 SELECT 作用域内查找，避免 UNION 各分支的字段互相串场。
            possible = [
                table for table in select_scope(enclosing_select(column))
                if column_name.lower() in column_sets[table]
            ]
            if len(possible) != 1:
                raise ValueError(
                    f"字段 {column_name} 未限定表名或存在歧义：{possible}"
                )
            return possible[0], column_name

    # MySQL 允许 ORDER BY / GROUP BY / HAVING 直接引用 SELECT 的输出别名。
    # 这类标识符不是表字段，校验时必须跳过，否则会被当成"未限定的列"误判。
    select_aliases = {
        projection.alias.lower()
        for select in tree.find_all(exp.Select)
        for projection in select.expressions
        if projection.alias
    }

    def is_alias_reference(column: exp.Column) -> bool:
        """判断是否为 ORDER BY / GROUP BY / HAVING 中对 SELECT 别名的引用。"""
        # 带表限定的是真实字段；名字不在输出别名里，自然也不是别名引用。
        if column.table or column.name.lower() not in select_aliases:
            return False

        node = column.parent
        while node is not None:
            if isinstance(node, (exp.Order, exp.Group, exp.Having)):
                # 只认查询层的 ORDER BY；窗口函数 OVER(... ORDER BY) 无别名作用域。
                return isinstance(node.parent, exp.Select)
            if isinstance(node, exp.Select):
                return False
            node = node.parent
        return False

    # 检查 SELECT、WHERE、GROUP BY、ORDER BY 等处的字段。
    for column in tree.find_all(exp.Column):
        # t.* 或 * 不需要映射到某一个具体字段。
        if column.is_star:
            if column.table and column.table.lower() not in alias_to_table:
                raise ValueError(f"星号使用了未知表别名：{column.table}")
            continue
        if is_alias_reference(column):
            continue
        resolve_column(column)

    declared_edges = set()
    for relation in relations:
        declared_edges.add((
            relation["left_table"].lower(),
            relation["left_column"].lower(),
            relation["right_table"].lower(),
            relation["right_column"].lower(),
        ))
        declared_edges.add((
            relation["right_table"].lower(),
            relation["right_column"].lower(),
            relation["left_table"].lower(),
            relation["left_column"].lower(),
        ))

    for join in tree.find_all(exp.Join):
        if join.args.get("using"):
            raise ValueError("JOIN USING 不支持；请使用 rels.json 中的显式关联字段。")

        on_clause = join.args.get("on")
        if on_clause is None:
            raise ValueError("每个 JOIN 都必须有 ON 条件。")

        join_has_relation = False
        for equality in on_clause.find_all(exp.EQ):
            left, right = equality.this, equality.expression
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue

            left_table, left_column = resolve_column(left)
            right_table, right_column = resolve_column(right)

            edge = (
                left_table.lower(),
                left_column.lower(),
                right_table.lower(),
                right_column.lower(),
            )
            if edge not in declared_edges:
                raise ValueError(
                    "JOIN 使用了 rels.json 未声明的字段关系："
                    f"{left_table}.{left_column} = "
                    f"{right_table}.{right_column}"
                )
            join_has_relation = True

        if not join_has_relation:
            raise ValueError("JOIN ON 中没有 rels.json 声明的关联条件。")


# ----------------------------
# Agent Tools
# ----------------------------

def tool_intent(state: AgentState) -> None:
    log("Tool / 意图识别：提取主题、指标、筛选条件、粒度和排序。")

    state.intent = call_llm(
        system_prompt=(
            "你是数据库问数 Agent 的意图识别器。"
            "将问题拆成简洁、可供选表和生成 SQL 使用的结构化要素。"
            "不要猜测表名或字段名。\n"
            "filters 只放【独立于实体的筛选条件】，必须是对象数组，每项包含：\n"
            '  {"field":"条件的业务含义","operator":"= | IN | > | < | LIKE | BETWEEN | IS NOT NULL",'
            '"value":..., "source_text":"原文片段"}\n'
            "规则：\n"
            "1. 若“对公/个人/集团/同业”等限定词与业务对象构成同一个名词短语"
            "（如“对公贷款借据”“对公客户”“个人存款账户”），属于【实体限定】，"
            "应完整写进 entities，不要拆出来放进 filters。\n"
            "2. 只有时间范围、金额阈值、状态、机构、证件号/编号、产品等独立条件才进 filters。\n"
            "3. filters 不是必填项；没有独立条件就返回空数组 []。\n"
            "4. field 只写业务含义，不要编造字段名；无法落地时应保留业务含义而不是硬套。\n"
            "只返回 JSON：intent、entities、metrics、filters、time_scope、grain、sort、notes"
        ),
        user_payload={"question": state.question},
    )
    state.phase = "select_tables"
    log(f"Observation / 意图结果：{json.dumps(state.intent, ensure_ascii=False)}")


def tool_select_tables(state: AgentState) -> None:
    log("Tool / Schema Linking：读取轻量表目录，让 LLM 选候选表。")
    db = MySQLTools(MYSQL_CONFIG)
    state.catalog = db.get_catalog()

    chosen = call_llm(
        system_prompt=(
            "你是数据库 Schema Linking 模块。"
            "根据用户问题、意图结果和表名/表注释，选出可能需要的表。"
            "只能从 catalog 中选择表名，不要虚构名称。"
            "若有不确定性，可返回多个候选表。只返回 JSON："
            '{"tables":["表名"],"reason":"简短原因"}'
        ),
        user_payload={
            "question": state.question,
            "intent": state.intent,
            "catalog": state.catalog,
        },
    )

    allowed = {item["table"] for item in state.catalog}
    state.selected_tables = [
        name for name in chosen.get("tables", [])
        if name in allowed
    ]
    if not state.selected_tables:
        raise RuntimeError("LLM 没有选出有效的候选表。")

    state.phase = "load_schema"
    log(
        "Observation / 候选表："
        + ", ".join(state.selected_tables)
        + f"；原因：{chosen.get('reason', '')}"
    )


def tool_load_schema(state: AgentState) -> None:
    log("Tool / Schema：只加载候选表的字段和字段注释，并读取 rels.json。")
    db = MySQLTools(MYSQL_CONFIG)
    state.schemas = db.get_table_schema(state.selected_tables)

    all_relations = load_relations(RELS_PATH)
    chosen = {name.lower() for name in state.selected_tables}
    state.relations = [
        relation
        for relation in all_relations
        if relation["left_table"].lower() in chosen
        and relation["right_table"].lower() in chosen
    ]

    state.phase = "generate_sql"
    log(
        f"Observation / 已加载 {len(state.schemas)} 张表、"
        f"{sum(map(len, state.schemas.values()))} 个字段、"
        f"{len(state.relations)} 条候选关系。"
    )


def tool_generate_sql(state: AgentState) -> None:
    log("Tool / SQL Generation：根据问题、意图、Schema 和真实关系生成 SQL。")

    response = call_llm(
        system_prompt=(
            "你是 MySQL SQL 生成器。只生成一条只读 SELECT 查询。"
            "只能使用提供的表和字段。"
            "JOIN 条件只能使用提供的 relationships 中的关系。"
            "禁止虚构表、字段、关联关系。"
            "如果信息不足，返回空 SQL 并在 explanation 中说明。"
            "只返回 JSON：{\"sql\":\"...\",\"explanation\":\"...\"}"
        ),
        user_payload={
            "question": state.question,
            "intent": state.intent,
            "schemas": state.schemas,
            "relationships": state.relations,
            "previous_validation_error": state.validation_error,
            "previous_execution_error": state.execution_error,
        },
    )

    state.sql = str(response.get("sql", "")).strip()
    if not state.sql:
        raise RuntimeError(
            response.get("explanation") or "LLM 未能生成 SQL。"
        )

    state.validation_error = ""
    state.execution_error = ""
    state.phase = "validate_sql"
    log(f"Observation / SQL：\n{state.sql}")
    if response.get("explanation"):
        log(f"SQL 说明：{response['explanation']}")


def tool_validate_sql(state: AgentState) -> None:
    log("Tool / SQL 校验：解析 SQL，并检查表、字段和 JOIN。")
    validate_sql(state.sql, state.schemas, state.relations)
    state.phase = "execute_sql"
    log("Observation / SQL 校验通过。")


def tool_execute_sql(state: AgentState) -> None:
    log("Tool / SQL Execution：在 MySQL 执行只读查询。")
    db = MySQLTools(MYSQL_CONFIG)
    state.result = db.execute_select(state.sql)
    state.phase = "finish"
    log(f"Observation / 查询完成，返回 {len(state.result)} 行。")


# ----------------------------
# Planner + Agent Loop
# ----------------------------

def planner(state: AgentState) -> str:
    """根据当前 State 决定下一步 Action。"""
    if state.phase == "intent":
        return "intent"
    if state.phase == "select_tables":
        return "select_tables"
    if state.phase == "load_schema":
        return "load_schema"
    if state.phase == "generate_sql":
        return "generate_sql"
    if state.phase == "validate_sql":
        return "validate_sql"
    if state.phase == "execute_sql":
        return "execute_sql"
    if state.phase == "finish":
        return "finish"
    return "finish"


def run_agent(question: str) -> AgentState:
    state = AgentState(question=question)
    actions = {
        "intent": tool_intent,
        "select_tables": tool_select_tables,
        "load_schema": tool_load_schema,
        "generate_sql": tool_generate_sql,
        "validate_sql": tool_validate_sql,
        "execute_sql": tool_execute_sql,
    }

    log(f"收到问题：{question}")

    while not state.done:
        action = planner(state)
        log(f"Planner / 当前阶段={state.phase}，选择 Action={action}")

        if action == "finish":
            state.done = True
            break

        try:
            actions[action](state)
        except Exception as exc:
            message = str(exc)
            log(f"Observation / 出错：{message}")

            if action == "validate_sql":
                state.validation_error = message
            elif action == "execute_sql":
                state.execution_error = message
            else:
                state.done = True
                raise

            if state.repair_attempts < MAX_REPAIR_ATTEMPTS:
                state.repair_attempts += 1
                state.phase = "generate_sql"
                log(
                    "Planner / 将错误反馈给 SQL Generator，"
                    f"开始第 {state.repair_attempts} 次修复。"
                )
            else:
                state.phase = "finish"
                log("Planner / 修复次数已用完，结束。")

    return state


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("请输入你的数据库问题：").strip()

    if not question:
        print("问题不能为空。")
        return

    state = run_agent(question)

    print("\n========== 最终结果 ==========")
    if state.sql:
        print("SQL:")
        print(state.sql)

    if state.result:
        print(f"\n结果（最多显示 {MAX_RESULT_ROWS} 行）：")
        print(json.dumps(state.result, ensure_ascii=False, indent=2))
    elif state.phase == "finish":
        print("本次没有返回查询结果；请查看上方错误信息。")


if __name__ == "__main__":
    main()