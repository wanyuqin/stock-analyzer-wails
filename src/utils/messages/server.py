import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Any, Dict, Literal

# Message Types
MESSAGE_TYPE_ANSWER = "answer"
MESSAGE_TYPE_THINKING = "thinking"
MESSAGE_TYPE_TOOL_REQUEST = "tool_request"
MESSAGE_TYPE_TOOL_RESPONSE = "tool_response"
MESSAGE_TYPE_MESSAGE_START = "message_start"
MESSAGE_TYPE_MESSAGE_END = "message_end"
MESSAGE_TYPE_ERROR = "error"



MessageType = Literal[
    MESSAGE_TYPE_ANSWER,
    MESSAGE_TYPE_THINKING,
    MESSAGE_TYPE_TOOL_REQUEST,
    MESSAGE_TYPE_TOOL_RESPONSE,
    MESSAGE_TYPE_MESSAGE_START,
    MESSAGE_TYPE_MESSAGE_END,
    MESSAGE_TYPE_ERROR,
]


# Message End Codes
MESSAGE_END_CODE_SUCCESS = "0"
MESSAGE_END_CODE_CANCELED = "1"

# Tool Response Codes
TOOL_RESP_CODE_SUCCESS = "0"


@dataclass
class TokenCost:
    input_tokens: int = field(default_factory=int)
    output_tokens: int = field(default_factory=int)
    total_tokens: int = field(default_factory=int)


@dataclass
class MessageEndDetail:
    code: str = field(default_factory=str)  # 错误码
    message: str = field(default_factory=str)  # 错误消息

    token_cost: Optional[TokenCost] = field(default=None)  # 消耗的token数量
    time_cost_ms: Optional[int] = field(default=None)  # 耗时，单位毫秒


@dataclass
class MessageStartDetail:
    local_msg_id: str = field(default_factory=str)
    msg_id: str = field(default_factory=str)
    execute_id: str = field(default_factory=str)

@dataclass
class ErrorDetail:
    local_msg_id: str = field(default_factory=str)
    code: str = field(default_factory=str)  # 错误码
    error_msg: str = field(default_factory=str)  # 错误消息

@dataclass
class ToolRequestDetail:
    tool_call_id: str = field(default_factory=str)
    tool_name: str = field(default_factory=str)
    parameters: Dict[str, Any] = field(default_factory=dict)  # tool_name to parameters


@dataclass
class ToolResponseDetail:
    tool_call_id: str = field(default_factory=str)

    code: str = field(default_factory=str)  # 错误码
    message: str = field(default_factory=str)  # 错误消息

    result: str = field(default_factory=str)  # tool执行结果
    time_cost_ms: Optional[int] = field(default=None)  # 耗时，单位毫秒


@dataclass
class ServerMessageContent:
    answer: Optional[str] = field(default=None)  # 回答内容
    thinking: Optional[str] = field(default=None)  # 思考内容
    tool_request: Optional[ToolRequestDetail] = field(default=None)  # tool请求详情
    tool_response: Optional[ToolResponseDetail] = field(default=None)  # tool响应详情

    error: Optional[ErrorDetail] = field(default=None)  # 错误详情

    message_start: Optional[MessageStartDetail] = field(default=None)  # 消息开始详情, 接收到消息后发送
    message_end: Optional[MessageEndDetail] = field(default=None)      # 消息结束详情, 处理完消息后发送


@dataclass
class ServerMessage:
    type: MessageType = field(default_factory=str)  # 消息类型
    session_id: str = field(default_factory=str)  # 会话id
    query_msg_id: str = field(default_factory=str)  # 对应的 client_msg_id
    reply_id: str = field(default_factory=str)  # 回复id, 一次回复过程中唯一
    msg_id: str = field(
        default_factory=str
    )  # 消息 id, 每个单独的消息(tool_request/tool_response等)都有一个唯一的 msg_id
    sequence_id: int = field(default_factory=int)  # 消息在回复中的序号, 从1开始递增
    finish: bool = field(
        default_factory=bool
    )  # 如果是流式消息,比如流式的 thinking,表示是否是最后一条消息
    content: ServerMessageContent = field(
        default_factory=ServerMessageContent
    )  # 消息内容
    log_id: str = field(default_factory=str)  # 日志id, 用于关联日志

    def dict(self):
        return asdict(self)



def create_message_end_dict(
    code: str,
    message: str,
    session_id: str,
    query_msg_id: str,
    log_id: str,
    time_cost_ms: int,
    reply_id: str = '',
    sequence_id: int = 1,
) -> Dict[str, Any]:
    """创建 message_end 消息字典，复用现有的 ServerMessage 结构"""
    return ServerMessage(
        type=MESSAGE_TYPE_MESSAGE_END,
        session_id=session_id,
        query_msg_id=query_msg_id,
        reply_id=reply_id,
        msg_id=str(uuid.uuid4()),
        sequence_id=sequence_id,
        finish=True,
        content=ServerMessageContent(
            message_end=MessageEndDetail(
                code=code,
                message=message,
                time_cost_ms=time_cost_ms,
                token_cost=TokenCost(input_tokens=0, output_tokens=0, total_tokens=0),
            )
        ),
        log_id=log_id,
    ).dict()


def create_message_error_dict(
    code: str,
    message: str,
    session_id: str,
    query_msg_id: str,
    log_id: str,
    reply_id: str = '',
    sequence_id: int = 1,
    local_msg_id: str = '',
) -> Dict[str, Any]:
    return ServerMessage(
        type=MESSAGE_TYPE_ERROR,
        session_id=session_id,
        query_msg_id=query_msg_id,
        reply_id=reply_id,
        sequence_id=sequence_id,
        finish=True,
        content=ServerMessageContent(
            error=ErrorDetail(
                local_msg_id=local_msg_id,
                code=code,
                error_msg=message,
            )
        ),
        log_id=log_id,
    ).dict()
