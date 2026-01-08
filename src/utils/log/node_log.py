import time
import logging
from uuid import UUID
from openai import BaseModel
from utils.log.config import LOG_DIR
from utils.log.common import get_execute_mode, is_prod
import uuid
from langchain_core.callbacks import BaseCallbackHandler
from coze_coding_utils.runtime_ctx.context import Context
import os
import sys
import json
from typing import Dict, Optional, Any
from pydantic import BaseModel
from utils.log.parser import LangGraphParser
import asyncio


class ParamInfo:
    name: str  # 参数名
    ptype: str  # 参数类型
    optional: bool  # 是否选填
    description: Optional[str] = None  # 参数说明
    items: Optional['ParamInfo'] = None  # 子参数，用于数组
    default: Optional[Any] = None  # 默认值，用于基础类型


# 2. 确保日志目录存在
# 尝试使用可写目录，先尝试/app目录，如果失败则使用/tmp目录
try:
    LOG_FILE = os.path.join(LOG_DIR, 'app.log')
    # 测试写入权限
    with open(LOG_FILE, 'a') as f:
        pass
except Exception as e:
    # 如果无法在/app目录写入，则使用/tmp目录
    # 确保创建的是目录而不是文件路径
    FALLBACK_LOG_DIR = '/tmp/work/logs/bypass'
    os.makedirs(FALLBACK_LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(FALLBACK_LOG_DIR, 'app.log')
    print(f"Warning: Using fallback log directory: {FALLBACK_LOG_DIR}, due to error: {e}", flush=True)

# 3. 只配置控制台输出的logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# 获取logger实例（仅用于控制台输出）
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_log(log_entry):
    """
    直接使用文件操作写入JSON格式日志，确保立即刷新到磁盘
    :param log_entry: 符合要求格式的日志字典
    """
    try:
        if is_prod():
            #  线上不打日志，待具备清理能后再打
            return None
        log_json = json.dumps(log_entry, ensure_ascii=False)

        # 修改为行缓冲模式（buffering=1）而不是无缓冲模式
        with open(LOG_FILE, 'a', encoding='utf-8', buffering=1) as f:  # 行缓冲模式
            f.write(log_json + '\n')
            # 显式调用flush和fsync确保数据写入磁盘
            f.flush()
            os.fsync(f.fileno())

        # 同时输出到控制台以便调试
        level = log_entry.get('level', 'info').lower()
        log_method = getattr(logger, level, logger.info)
        log_method(log_entry.get('message', ''))

    except Exception as e:
        # 如果写入失败，打印到标准错误
        print(f"Failed to write log: {e}", flush=True)
        # 尝试再次写入作为备选
        try:
            log_json = json.dumps(log_entry, ensure_ascii=False)
            print(f"Attempting fallback write: {log_json}", flush=True)
            # 修改备选方案为行缓冲模式
            f = open(LOG_FILE, 'a', encoding='utf-8', buffering=1)
            try:
                f.write(log_json + '\n')
                f.flush()
                os.fsync(f.fileno())
            finally:
                f.close()
        except Exception as fallback_e:
            print(f"Fallback log write failed: {fallback_e}", flush=True)


def create_log_entry(level="info", message="", timestamp=None, log_id=None, latency=0,
                     input_data="", output_data="", node_id="", project_id="", commit_id="",
                     execute_mode="run", caller="", node_type="", node_title="",
                     token="", cost="", error_code="", error_message="", event_type="", execution_id="", node_name="",
                     method=""):
    """
    创建符合要求的日志条目
    :param level: 日志级别
    :param message: 日志内容
    :param timestamp: 时间戳，默认为当前时间
    :param log_id: 日志ID，默认为生成的UUID
    :param latency: 延迟时间
    :param input_data: 输入数据
:param output_data: 输出数据
    :param node_id: 节点ID
    :param project_id: 项目ID
    :param commit_id: 提交ID
    :param execute_mode: 执行模式
    :param caller: 调用者
    :param node_type: 节点类型
    :param node_title: 节点标题
    :param token: token信息
    :param cost: 成本信息
    :param error_code: 错误码
    :param error_message: 错误信息
    :param event_type: 事件类型
    :param execution_id: 执行唯一ID（可选）
    :param node_name: 节点名称
    :param method: 方法名称
    :return: 格式化的日志字典
    """
    if timestamp is None:
        timestamp = int(time.time() * 1000)
    # 限制input_data和output_data的长度，设置为1MB，则使用兜底文案
    if len(input_data) > 1024 * 1024:
        input_data = "输入数据长度超过1MB，已截断"
    if len(output_data) > 1024 * 1024:
        output_data = "输出数据长度超过1MB，已截断"
    return {
        "level": level,
        "message": message,
        "timestamp": timestamp,
        "log_id": log_id,
        "latency": latency,
        "input": input_data,
        "output": output_data,
        "node_id": node_id,
        "project_id": project_id,
        "commit_id": commit_id,
        "execute_mode": execute_mode,
        "caller": caller,
        "node_type": node_type,
        "node_title": node_title,
        "token": token,
        "cost": cost,
        "error_code": error_code,
        "error_message": error_message,
        "type": event_type,
        "execute_id": execution_id,
        "node_name": node_name,
        "method": method,
    }


def log_workflow_start(project_id, commit_id, log_id=None, execute_id="", input_data="", method=""):
    """
    记录流程开始日志
    :param project_id: 项目ID
    :param commit_id: 提交ID
    :param is_test_run: 是否试运行
    :param log_id: 日志ID（可选）
    :param execute_id: 执行唯一ID（可选）
    """
    event_type = "test_run_start" if not is_prod() else "run_start"
    execute_mode = "test_run" if not is_prod() else "run"

    message = f"Workflow started - {'Test Run' if is_prod() else 'Run'}"

    log_entry = create_log_entry(
        level="info",
        message=message,
        log_id=log_id,
        project_id=project_id,
        commit_id=commit_id,
        execute_mode=execute_mode,
        event_type=event_type,
        execution_id=execute_id,
        input_data=input_data,
        method=method,
    )

    write_log(log_entry)


def log_workflow_end(execution_id, output=None, total_time=None, status="success", token_consumed=None,
                     error_reason=None, error_code=None, is_test_run=False, log_id="", method=""):
    """
    记录流程结束日志
    :param execution_id: 执行唯一ID
    :param output: 流程输出
    :param total_time: 流程耗时
    :param status: 流程执行状态
    :param token_consumed: 流程token消耗
    :param error_reason: 错误原因
    :param error_code: 错误码
    :param is_test_run: 是否试运行
    """
    level = "error" if status == "error" else "info"
    execute_mode = "test_run" if is_test_run else "run"

    message = f"Workflow completed - {status}"
    if execution_id:
        message += f" (ID: {execution_id})"

    log_entry = create_log_entry(
        level=level,
        message=message,
        latency=int(total_time * 1000) if total_time else 0,
        output_data=_serialize_data(output),
        execute_mode=execute_mode,
        event_type="test_run_done" if is_test_run else "done",
        token=str(token_consumed) if token_consumed else "",
        error_code=str(error_code) if error_code else "",
        error_message=str(error_reason) if error_reason else "",
        execution_id=execution_id,
        log_id=log_id,
        method=method,
    )

    write_log(log_entry)


class Logger(BaseCallbackHandler):
    def __init__(self, graph, ctx: Context):
        self.root_run_id = None
        self.graph = graph
        self.runtime_ctx = ctx
        self.start_time = time.time()
        self.parser = LangGraphParser(graph)

    run_id_map: Dict[uuid.UUID, str] = {}

    def on_chain_start_graph(
            self,
            serialized: dict[str, Any],
            inputs: dict[str, Any],
            *,
            run_id: uuid.UUID,
            parent_run_id: uuid.UUID | None = None,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
    ) -> Any:
        if metadata is None:
            metadata = {}
        node_name_value = kwargs.get("name")
        node_name: str | None = node_name_value if isinstance(node_name_value, str) else None
        if node_name:
            self.run_id_map[run_id] = node_name
        if parent_run_id is None:
            self._on_graph_start(inputs)  # workflow 开始
        node_info = self.parser.nodes.get(node_name) if node_name is not None else None
        if node_info is None:
            # 检查是否为条件节点
            if node_name in self.parser.condition_funcs:
                # 记录条件节点日志
                log_entry = create_log_entry(
                    level="info",
                    message=f"Condition node '{node_name}' started",
                    input_data=_serialize_data(inputs),
                    node_name=self.parser.condition_funcs[node_name]["cond_node_name"],  # 前端的条件节点名
                    execution_id=self.runtime_ctx.run_id,
                    execute_mode=get_execute_mode(),
                    event_type="node_start",
                    log_id=self.runtime_ctx.logid,
                    method=self.runtime_ctx.method,
                    node_type="condition"
                )
                write_log(log_entry)
                return
            logger.debug(f"Node {node_name} not found in graph")
            return
        log_entry = create_log_entry(
            level="info",
            message=f"Node '{node_info.name}' started",
            input_data=_serialize_data(inputs),
            node_id=node_info.node_id,
            node_type=node_info.node_type,
            node_title=node_info.title,
            execute_mode=get_execute_mode(),
            event_type="node_start",
            execution_id=self.runtime_ctx.run_id,
            log_id=self.runtime_ctx.logid,
            node_name=node_info.name,
            method=self.runtime_ctx.method,
        )
        write_log(log_entry)

    def on_chain_end_graph(
            self,
            outputs: dict[str, Any],
            *,
            run_id: uuid.UUID,
            parent_run_id: uuid.UUID | None = None,
            **kwargs: Any,
    ) -> Any:
        node_name = self.run_id_map.pop(run_id, None)
        if parent_run_id is None:  # 根节点
            self._on_graph_end(outputs)
        elif node_name:
            # Node end
            node_info = self.parser.nodes.get(node_name, None)
            if node_info is None:
                # 检查是否为条件节点
                if node_name in self.parser.condition_funcs:
                    # 记录条件节点日志
                    log_entry = create_log_entry(
                        level="info",
                        message=f"Condition node '{node_name}' ended",
                        output_data=_serialize_data(outputs),
                        node_name=self.parser.condition_funcs[node_name]["cond_node_name"],  # 前端的条件节点名
                        execution_id=self.runtime_ctx.run_id,
                        execute_mode=get_execute_mode(),
                        event_type="node_end",
                        log_id=self.runtime_ctx.logid,
                        method=self.runtime_ctx.method,
                        node_type="condition"
                    )
                    write_log(log_entry)
                    return
                logger.debug(f"Node {node_name} not found in graph")
                return
            log_entry = create_log_entry(
                level="info",
                message=f"Node '{node_info.name}' ended",
                output_data=_serialize_data(outputs),
                node_id=node_info.node_id,  # 注册的时候使用的function name，前端用来流转
                node_type=node_info.node_type,
                node_title=node_info.title,
                execute_mode=get_execute_mode(),
                event_type="node_end",
                execution_id=self.runtime_ctx.run_id,
                log_id=self.runtime_ctx.logid,
                node_name=node_info.name,
                method=self.runtime_ctx.method,
            )
            write_log(log_entry)

    def _on_graph_start(self, inputs: Dict[str, Any]):
        # Workflow start
        project_id = os.getenv("COZE_PROJECT_ID", "")
        commit_id = ""  # This might need to be sourced from metadata if available
        log_workflow_start(
            project_id=project_id,
            commit_id=commit_id,
            log_id=str(self.runtime_ctx.logid),
            execute_id=self.runtime_ctx.run_id,
            input_data=_serialize_data(inputs),
            method=self.runtime_ctx.method,
        )

    def _on_graph_end(self, outputs: Dict[str, Any]):
        # Workflow end
        total_time = time.time() - self.start_time
        log_workflow_end(
            execution_id=self.runtime_ctx.run_id,
            output=outputs,
            total_time=total_time,
            status="success",
            log_id=self.runtime_ctx.logid,
            is_test_run=not is_prod(),
            method=self.runtime_ctx.method,
        )

    def on_chain_error(
            self,
            error: BaseException,
            *,
            run_id: UUID,
            parent_run_id: UUID | None = None,
            **kwargs: Any,
    ) -> Any:
        event_type = "error"

        # 如果是取消操作，事件类型改为取消
        if isinstance(error, asyncio.CancelledError):
            logger.info(f"Task cancelled for run_id: {run_id}")
            event_type = "cancel"
        # 记录节点失败日志
        node_name = self.run_id_map.pop(run_id, "")
        # Node end
        node_id = ""
        node_title = ""
        node_type = ""
        node_info = self.parser.nodes.get(node_name) if node_name is not None else None
        if node_info is not None:
            node_name = node_info.name if node_info else ""
            node_title = node_info.title if node_info else ""
            node_id = node_info.node_id if node_info else ""
            node_type = node_info.node_type if node_info else ""
        # 节点异常
        error_log_entry = create_log_entry(
            level="error",
            message=f"Workflow {node_id} ended with error",
            node_id=node_id,
            node_type=node_type,
            node_title=node_title,
            execute_mode=get_execute_mode(),
            event_type=event_type,
            execution_id=self.runtime_ctx.run_id,
            log_id=self.runtime_ctx.logid,
            error_message=str(error),
            node_name=node_name,
            method=self.runtime_ctx.method,
        )
        write_log(error_log_entry)

    def get_node_tags(self, node_name: str) -> dict[str, str]:
        node_tags = {}
        if node_name is None or node_name == "":
            return node_tags

        node_info = self.parser.nodes.get(node_name, None)
        if node_info is None:
            logger.debug(f"Node {node_name} not found in graph")
            return {}
        node_tags["node_id"] = node_info.node_id
        node_tags["node_type"] = self.parser.get_node_type(node_info.node_id)
        node_tags["node_title"] = node_info.title
        node_tags["node_name"] = node_info.name
        return node_tags

    def get_node_name(self, node_name: str) -> str:
        # 获取node title
        if node_name == "LangGraph":
            return "Workflow"
        node_info = self.parser.nodes.get(node_name, None)
        if node_info is None:
            logger.debug(f"Node {node_name} not found in graph")
            return node_name
        node_title = node_info.title if node_info else node_name  # 没有title时，使用node_name
        return node_title


def _serialize_data(data: Any) -> str:
    """
    增强版数据序列化函数，支持：
    - Pydantic BaseModel
    - 字典/列表等基础类型
    - 自定义对象（通过 __dict__ 序列化）
    - 特殊字符（保证 ASCII 编码）
    """

    def _recursive_serialize(item: Any):
        """递归序列化单个元素"""
        # 处理 Pydantic 模型
        if isinstance(item, BaseModel):
            return item.model_dump()  # 先转字典再序列化

        # 处理列表/元组
        elif isinstance(item, (list, tuple)):
            return [_recursive_serialize(sub_item) for sub_item in item]

        # 处理字典
        elif isinstance(item, dict):
            return {key: _recursive_serialize(value) for key, value in item.items()}

        # 处理自定义对象（有 __dict__ 属性的）
        elif hasattr(item, '__dict__') and not isinstance(item, (str, int, float, bool, type(None))):
            return _recursive_serialize(item.__dict__)

        # 基础类型直接返回
        else:
            return item

    try:
        # 先递归处理数据为可序列化的基础类型
        serialized_data = _recursive_serialize(data)
        # 最终序列化为 JSON 字符串
        return json.dumps(serialized_data, ensure_ascii=False, indent=None)

    except Exception as e:
        logger.error(f"Error serializing data: {e}", exc_info=True)
        # 降级处理：返回字符串表示
        if len(str(data)) > 1000:
            # 避免bytes类型返回过大，打挂线程
            logger.info(f"Data too large and truncated, len={len(str(data))}")
            return ""
        return str(data)
