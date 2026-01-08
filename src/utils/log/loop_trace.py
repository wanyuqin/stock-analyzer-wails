import os
import cozeloop
from cozeloop.integration.langchain.trace_callback import LoopTracer
from langchain_core.runnables import RunnableConfig
from utils.log.common import get_execute_mode
from utils.log.node_log import Logger

space_id = os.getenv("COZE_PROJECT_SPACE_ID", "YOUR_SPACE_ID")
api_token = os.getenv("COZE_LOOP_API_TOKEN", "YOUR_LOOP_API_TOKEN")
base_url = os.getenv("COZE_LOOP_BASE_URL", "https://api.coze.cn")
commit_hash = os.getenv("COZE_PROJECT_COMMIT_HASH","") # 发布版本的hash值

cozeloopTracer = cozeloop.new_client(
    workspace_id=space_id,
    api_token=api_token,
    api_base_url=base_url,
)
cozeloop.set_default_client(cozeloopTracer)


def init_run_config(graph, ctx):
    tracer = Logger(graph, ctx)
    tracer.on_chain_start = tracer.on_chain_start_graph  # 非必须
    tracer.on_chain_end = tracer.on_chain_end_graph
    trace_callback_handler = LoopTracer.get_callback_handler(
        cozeloopTracer,
        add_tags_fn=tracer.get_node_tags,
        modify_name_fn=tracer.get_node_name,
        tags={
            "project_id": ctx.project_id,
            "execute_mode": get_execute_mode(),
            "log_id": ctx.logid,
            "commit_hash": commit_hash,
        }
    )
    config = RunnableConfig(
        callbacks=[
            tracer,
            trace_callback_handler
        ],
    )
    return config


def init_agent_config(graph, ctx):
    config = RunnableConfig(
        callbacks=[
            LoopTracer.get_callback_handler(
                cozeloopTracer,
                tags={
                    "project_id": ctx.project_id,
                    "execute_mode": get_execute_mode(),
                    "log_id": ctx.logid,
                    "commit_hash": commit_hash,
                }
                )
        ]
    )
    print("config", config)
    return config


# 保留add_trace_tags函数，作为对trace.set_tags的简单包装
def add_trace_tags(trace, tags):
    """
    为trace添加标签
    :param trace: trace对象
    :param tags: 标签字典
    """
    # 使用set_tags方法
    trace.set_tags(tags)

