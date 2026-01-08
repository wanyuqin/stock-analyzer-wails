import os
import inspect
import importlib
import ast
import textwrap
from pydantic import BaseModel
from typing import get_type_hints,Type,Optional,get_origin,Union,get_args
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import START, END


def get_graph_instance(module_name):
    module = importlib.import_module(module_name)
    for _, obj in inspect.getmembers(module):
        if isinstance(obj, CompiledStateGraph):
            return obj
    return None

def get_agent_instance(module_name, ctx):
    module = importlib.import_module(module_name)
    return module.build_agent(ctx)

# return: func, input_class, output_class
def get_graph_node_func_with_inout(graph, node_name):
    for node_id, node in graph.nodes.items():
        if node_id == START or node_id == END:
            continue

        if node.data:
            _func = node.data.func
            if _func.__name__ != node_name:
                continue

            # 获取函数签名
            sig = inspect.signature(_func)
            # 获取参数列表
            params = list(sig.parameters.values())
            input_cls = None
            if params:
                input_cls = params[0].annotation

            output_cls = ParamExtractHelper.get_concrete_return_class(_func)

            return _func, input_cls, output_cls

    return None, None, None

def is_agent_proj() -> bool:
    return os.getenv("COZE_PROJECT_TYPE", "workflow") == "agent"

def is_dev_env() -> bool:
    return os.getenv("COZE_PROJECT_ENV", "") == "DEV"


class ParamExtractHelper:
    @classmethod
    def get_concrete_return_class(cls, func) -> Optional[Type[BaseModel]]:
        """
        获取函数的返回类型,提取顺序
        1. Type Hint
        2. ast源码解析,支持情况
            2.1 函数被装饰器包裹
            2.2 return xx(id=x)
            2.3 return pkg_name.xx(id=x)
            2.4 返回变量 return some_var, 只查找函数内的赋值语句
        """

        original_func = inspect.unwrap(func)

        # 1. type hints
        output_cls = cls._extract_model_from_hints(original_func)

        # 2. 如果没有type hints，或者是泛型BaseModel, 通过AST解析
        if output_cls is None or (output_cls is BaseModel):
            print(f"Type hint insufficient for {original_func.__name__}, trying AST analysis...")
            ast_cls = cls._extract_model_from_ast(original_func)
            if ast_cls:
                output_cls = ast_cls

        # 3. 最终校验和输出
        if output_cls and issubclass(output_cls, BaseModel):
            return output_cls

        return None

    @classmethod
    def _extract_model_from_hints(cls, func)->Optional[Type[BaseModel]]:
        """从类型标注中提取 Pydantic 模型类"""
        try:
            hints = get_type_hints(func)
            return_type= hints.get('return', None)
            if return_type is None:
                return None

            # 处理 Optional[T] 或 Union[T, None] 的情况
            origin = get_origin(return_type)
            if origin is Union:
                args = get_args(return_type)
                # 过滤掉 NoneType
                valid_args = [arg for arg in args if arg is not type(None)]
                if len(valid_args) == 1:
                    return_type = valid_args[0]

            # 检查是否是类且是 BaseModel 的子类
            if isinstance(return_type, type) and issubclass(return_type, BaseModel):
                return return_type
        except Exception as e:
            print(f"Error extracting hints: {e}")

        return None

    @classmethod
    def _extract_model_from_ast(cls, func) -> Optional[Type[BaseModel]]:
        """
        解析函数源码，寻找 'return SomeClass()' 语句，并从函数上下文中找到对应的类。
        """
        try:
            # 获取源码并去缩进 (处理类方法或嵌套函数的情况)
            src = textwrap.dedent(inspect.getsource(func))
            tree = ast.parse(src)
            func_def = tree.body[0]

            # 遍历函数体寻找 return 语句
            # 注意：这里只简单的找最后一个 return 或者所有 return，
            # 复杂的逻辑流（多个不同 return）可能需要更复杂的处理
            return_node = None
            for node in ast.walk(func_def):
                if isinstance(node, ast.Return):
                    return_node = node
                    break

            if not return_node or not return_node.value:
                return None

            return cls._extract_model_from_ast_node(return_node.value, func)
        except Exception as e:
            print(f"Error extracting hints: {e}")
            pass

        return None

    @classmethod
    def _extract_model_from_ast_node(cls, node, func) -> Optional[Type[BaseModel]]:
        # 检查 return 的是不是一个函数调用 (例如 SummaryOutput())
        # 结构通常是: return SummaryOutput(id='SummaryOutput')
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                class_name = node.func.id
                # 查找全局命名空间
                globals_dict = func.__globals__
                if class_name in globals_dict:
                    possible_cls = globals_dict[class_name]
                    if isinstance(possible_cls, type) and issubclass(possible_cls, BaseModel):
                        return possible_cls

                # 查找闭包
                if func.__closure__:
                    for cell in func.__closure__:
                        if hasattr(cell.cell_contents, '__name__') and cell.cell_contents.__name__ == class_name:
                            possible_cls = cell.cell_contents
                            if isinstance(possible_cls, type) and issubclass(possible_cls, BaseModel):
                                return possible_cls
            elif isinstance(node.func, ast.Attribute):
                # 处理 return module.ClassName() 的情况
                return cls._extract_class_from_attribute(node.func, func)
        elif isinstance(node, ast.Name):
            # 情况2: 返回变量 return some_var
            # 这里需要更复杂的分析来追踪变量的类型
            # 作为简化，我们可以查找函数内的赋值语句
            return cls._find_variable_type(node.id, func)

        return None

    @classmethod
    def _extract_class_from_attribute(cls, node: ast.Attribute, func) -> Optional[Type[BaseModel]]:
        """
        递归解析 AST 属性节点，从 globals 中找到对应的对象。
        处理如: module.submodule.ClassName
        """
        try:
            global_ns = func.__globals__
            # 1. 获取属性名 (例如 'SummaryOutput')
            attr_name = node.attr

            # 2. 获取前缀对象 (例如 'schemas' 或 'module.submodule')
            value_node = node.value

            prefix_obj = None

            if isinstance(value_node, ast.Name):
                # 基础情况：前缀是一个变量名 (例如 'schemas')
                # 从 globals 中找到这个模块/对象
                prefix_obj = global_ns.get(value_node.id)

            elif isinstance(value_node, ast.Attribute):
                # 递归情况：前缀依然是一个属性 (例如 'pkg.mod'.Class)
                prefix_obj = cls._extract_class_from_attribute(value_node, global_ns)

            if prefix_obj:
                # 从模块/对象中获取类
                target_cls = getattr(prefix_obj, attr_name, None)
                if isinstance(target_cls, type) and issubclass(target_cls, BaseModel):
                    return target_cls

        except Exception:
            pass
        return None


    @classmethod
    def _find_variable_type(cls, var_name: str, func) -> Optional[Type[BaseModel]]:
        """查找变量的类型（简化版本）"""
        # 这是一个复杂的问题，需要完整的控制流分析
        # 这里提供一个简化的实现，只查找直接赋值
        try:
            source = inspect.getsource(func)
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                    # 查找赋值语句
                    for stmt in node.body:
                        if (isinstance(stmt, ast.Assign) and
                            any(isinstance(target, ast.Name) and target.id == var_name
                                for target in stmt.targets)):
                            # 找到对目标变量的赋值
                            if isinstance(stmt.value, ast.Call):
                                return cls._extract_model_from_ast_node(stmt.value, func)
        except:
            pass
        return None
