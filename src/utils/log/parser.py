import inspect
from dataclasses import dataclass
from typing import Dict, Optional, Any, Callable, cast
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, END


# return: title, description, integrations
def extract_title_description(func_name, text: Optional[str]):
    title = func_name
    desc = ""
    integrations = []

    if text is None:
        return title, desc, integrations

    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith('title'):
            if line.startswith('title:'):
                title = line.split('title:', 1)[1].strip()
            if line.startswith('title：'):
                title = line.split('title：', 1)[1].strip()
        elif line.startswith('desc'):
            if line.startswith('desc:'):
                desc = line.split('desc:', 1)[1].strip()
            if line.startswith('desc：'):
                desc = line.split('desc：', 1)[1].strip()
        elif line.startswith("integrations"):
            items = ""
            if line.startswith('integrations:'):
                items = line.split('integrations:', 1)[1].strip()
            if line.startswith('integrations：'):
                items = line.split('integrations：', 1)[1].strip()
            for item in items.split(","):
                integrations.append(IntegrationInfo(title=item))

    if title == "":
        title = func_name

    return title, desc, integrations


@dataclass
class SourceLocation:
    file: str  # 节点定义源码文件路径
    line: int  # 节点定义源码行号


@dataclass
class ParamInfo:
    name: str  # 参数名
    ptype: str  # 参数类型
    optional: bool  # 是否选填
    description: Optional[str] = None  # 参数说明
    items: Optional['ParamInfo'] = None  # 子参数，用于数组
    default: Optional[Any] = None  # 默认值，用于基础类型


@dataclass
class IntegrationInfo:
    _id: str = ""  # 集成ID
    title: str = ""  # 集成名
    description: str = ""  # 集成描述


@dataclass
class NodeInfo:
    node_id: str  # langgraph中的node_id，可能是add_node添加或节点函数名同名
    name: str  # func name
    title: str
    description: str = ""
    node_type: str = ""


class LangGraphParser:
    def __init__(self, app: CompiledStateGraph):
        # 从LangGraph中获取图结构
        self.graph_app = app
        self.graph = app.get_graph()
        # 从图中构建节点信息
        self.nodes: Dict[str, NodeInfo] = {}  # NodeId -> NodeInfo
        # 构建基础信息 - 优先使用CompiledStateGraph中的信息
        self._build_node_info()
        self.condition_funcs = self._pre_process_conditional_fork_node_info()  # 跟踪condition节点的判断函数，因为中间会插入哑结点和condition节点

    def _is_agent_node(self, node_id: str) -> bool:
        """
        判断是否为Agent节点，当前是模型节点，通过add_node的metadata注入标记
        """
        node = self.graph.nodes.get(node_id)
        if node and node.metadata:
            return node.metadata.get('type', '').lower() == "agent"
        return False

    def _is_loop_node(self, node_id: str) -> bool:
        """判断是否为循环节点"""
        node = self.graph.nodes.get(node_id)
        if node and node.metadata:
            _type = node.metadata.get('type', '').lower()
            return _type == "looparray" or _type == "loopcond"

        return False

    def _is_looparray_node(self, node_id: str) -> bool:
        """判断是否为列表循环"""
        node = self.graph.nodes.get(node_id)
        if node and node.metadata:
            _type = node.metadata.get('type', '').lower()
            return _type == "looparray"

        return False

    def _is_loopcond_node(self, node_id: str) -> bool:
        """判断是否为条件循环"""
        node = self.graph.nodes.get(node_id)
        if node and node.metadata:
            _type = node.metadata.get('type', '').lower()
            return _type == "loopcond"

        return False

    def get_node_metadata(self, func_name: str) -> dict:
        node_id = ""
        for _, node in self.nodes.items():
            if node.name == func_name:
                node_id = node.node_id

        node = self.graph.nodes.get(node_id)
        if node and node.metadata:
            return node.metadata

        return {}

    def find_conditional_nodes(self):
        conditional_nodes = set()
        # 构建入边映射：target -> [sources]
        incoming = {}
        for edge in self.graph.edges:
            incoming.setdefault(edge.target, []).append(edge.source)

        # 遍历所有条件边，并将条件源映射到真实业务节点
        for edge in self.graph.edges:
            if getattr(edge, "conditional", False):
                src = edge.source
                node_obj = self.graph.nodes.get(src)
                # 源是哑节点（编译插入，无数据函数），回溯到它的上游真实节点
                if not node_obj or not getattr(node_obj, "data", None):
                    for parent in incoming.get(src, []):
                        parent_obj = self.graph.nodes.get(parent)
                        if parent_obj and getattr(parent_obj, "data", None):
                            conditional_nodes.add(parent)
                else:
                    conditional_nodes.add(src)
        return conditional_nodes

    def get_node_type(self, node_id):
        # 简单推断逻辑
        if node_id == START: return "start"
        if node_id == END: return "end"
        if self._is_loop_node(node_id): return "loop"
        if self._is_agent_node(node_id): return "agent"
        return "task"

    def _generate_node_title(self, node_name: str) -> str:
        """生成节点标题"""
        if node_name == START:
            return "开始"
        elif node_name == END:
            return "结束"
        else:
            return node_name

    def _enhance_loop_node(self, canvas_node: Dict, node_info):
        """完善循环节点定义"""
        # 添加循环条件描述
        canvas_node["definition"]["info"]["condition_summary"] = {}
        # TODO: 循环条件描述暂时使用功能描述
        if self._is_looparray_node(node_info.node_id):
            canvas_node["definition"]["info"]["looptype"] = "looparray"
            canvas_node["definition"]["info"]["condition_summary"]["looparray"] = node_info.description
        else:
            canvas_node["definition"]["info"]["looptype"] = "loopcond"
            canvas_node["definition"]["info"]["condition_summary"]["loopcond"] = node_info.description

    def _build_node_info(self):
        """
        构建节点参数信息
        - 函数名
        - 函数docstring
        - 入参、出参
        - 源代码路径，行号

        兜底：AST不可用或解析失败，从CompiledStateGraph提取基本信息
        """
        self._build_node_info_by_langgraph()

    def _build_node_info_by_langgraph(self):
        for node_id, node in self.graph.nodes.items():
            if node_id == START:
                # 开始节点，构建工作流的入参
                input_cls = self.graph_app.get_input_schema()
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id,
                    name=START,
                    title="开始",
                    node_type="start",
                )
                continue

            if node_id == END:
                # 结束节点，构建工作流的出参
                output_cls = self.graph_app.get_output_schema()
                self.nodes[node_id] = NodeInfo(
                    node_id=node_id,
                    name=END,
                    title="结束",
                    node_type="end",
                )
                continue

            data = getattr(node, "data", None)
            if data:
                _func = getattr(data, "func", None)
                if _func is None and callable(data):
                    _func = cast(Callable[..., Any], data)
                if _func is None:
                    continue
                node_name = _func.__name__
                docstring = inspect.getdoc(_func)
                title, desc, integrations = extract_title_description(node_name, docstring)
                if node_id not in self.nodes:
                    self.nodes[node_id] = NodeInfo(
                        node_id=node_id,
                        name=node_name,
                        title=title,
                        description=desc,
                        node_type=self.get_node_type(node_id),
                    )

    def _pre_process_conditional_fork_node_info(self):
        '''
        构建条件节点的主节点信息，主要是描述，后续可扩展输入输出
        原因：LangGraph条件判断函数不是真实节点，而就是一个普通的函数
        '''

        '''defaultdict(<class 'dict'>, {'join': {'should_continue_processing': BranchSpec(path=should_continue_processing(tags=None, recurse=True, explode_args=False, func_accepts={}), ends={'中文描述分支1': 'add_item_len', '默认分支': 'add_default_item_len'}, input_schema=<class 'graphs.state.BranchJoinInput'>)}})'''
        branches = self.graph_app.builder.branches

        conditional_funcs = {}  # parent_id: key : {"func":func,"branch_start_node":}
        for parent_id, check in branches.items():
            for check_func_name, spec in check.items():
                conditional_funcs[check_func_name] = {
                    "cond_node_name": "cond_" + parent_id} # 拼成前端的条件节点名
        return conditional_funcs
