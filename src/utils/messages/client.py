from dataclasses import dataclass, field
from typing import List, Optional, Literal

MESSAGE_TYPE_QUERY = "query"
MessageType = Literal["query"]

BLOCK_TYPE_TEXT = "text"
BLOCK_TYPE_UPLOAD_FILE = "upload_file"

BlockType = Literal["text", "upload_file"]


@dataclass
class UploadFileBlockDetail:
    file_name: str = field(default_factory=str)
    file_path: str = field(default_factory=str)
    url: str = field(default_factory=str)


@dataclass
class PromptBlockContent:
    text: Optional[str] = field(default=None)
    upload_file: Optional[UploadFileBlockDetail] = field(default=None)


@dataclass
class PromptBlock:
    type: BlockType = field(default_factory=str)
    content: PromptBlockContent = field(default_factory=PromptBlockContent)


@dataclass
class QueryDetail:
    prompt: List[PromptBlock] = field(default_factory=list)


@dataclass
class ClientMessageContent:
    query: Optional[QueryDetail] = field(default=None)


@dataclass
class ClientMessage:
    type: MessageType = field(default_factory=str)
    project_id: str = field(default_factory=str)
    session_id: str = field(default_factory=str)
    local_msg_id: str = field(default_factory=str)
    content: ClientMessageContent = field(default_factory=ClientMessageContent)
