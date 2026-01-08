import uuid
import json
import os
from typing import Any, Dict, List, Tuple, Iterator
import time
from utils.file.file import File, FileOps, infer_file_category

from utils.messages.client import (
    ClientMessage,
    ClientMessageContent,
    QueryDetail,
    PromptBlock,
    PromptBlockContent,
    UploadFileBlockDetail,
)
from utils.messages.server import (
    ServerMessage,
    ServerMessageContent,
    ToolRequestDetail,
    ToolResponseDetail,
    MessageStartDetail,
    MessageEndDetail,
    TokenCost,
    MESSAGE_TYPE_MESSAGE_START,
    MESSAGE_TYPE_MESSAGE_END,
    MESSAGE_END_CODE_SUCCESS,
    MESSAGE_TYPE_ANSWER,
    MESSAGE_TYPE_TOOL_REQUEST,
    MESSAGE_TYPE_TOOL_RESPONSE,
)


def to_stream_input(msg: ClientMessage) -> Dict[str, Any]:
    content_parts = []
    if msg and msg.content and msg.content.query and msg.content.query.prompt:
        for block in msg.content.query.prompt:
            if block.type == "text" and block.content and block.content.text:
                content_parts.append({"type": "text", "text": block.content.text})
            elif (
                    block.type == "upload_file"
                    and block.content
                    and block.content.upload_file
            ):
                file_info = block.content.upload_file
                file_type , _ = infer_file_category(file_info.url)
                file_data = File(url=file_info.url, file_type=file_type)
                # check is image
                if file_data.file_type == "image":
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f'{file_data.url}'
                        }
                    )
                    content_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": file_info.url},
                        }
                    )
                # check is video
                elif file_data.file_type == "video":
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f'{file_data.url}'
                        }
                    )
                    content_parts.append(
                        {
                            "type": "video_url",
                            "video_url": {"url": file_info.url},
                        }
                    )
                # check is audio
                elif file_data.file_type == "audio":
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"audio url: {file_info.url}",
                        }
                    )
                else:
                    file_content = FileOps.extract_text(file_data)
                    content_parts.append(
                        {
                            "type": "text",
                            "text": f"file name:{file_info.file_name}, url: {file_info.url}\n\nFile Content:\n{file_content}",
                        }
                    )

    return {"messages": [{"role": "user", "content": content_parts}]}


def to_client_message(d: Dict[str, Any]) -> Tuple[ClientMessage, str]:
    prompt_list = d.get("content", {}).get("query", {}).get("prompt", [])
    blocks: List[PromptBlock] = []
    for b in prompt_list:
        c = b.get("content") or {}
        b_type = b.get("type", "text")

        if b_type == "text":
            text = c.get("text") if isinstance(c, dict) else None
            blocks.append(
                PromptBlock(
                    type=b_type, content=PromptBlockContent(text=text)
                )
            )
        elif b_type == "upload_file":
            upload_file_data = c.get("upload_file") if isinstance(c, dict) else None
            upload_file = None
            if upload_file_data:
                upload_file = UploadFileBlockDetail(
                    file_name=upload_file_data.get("file_name", ""),
                    file_path=upload_file_data.get("file_path", ""),
                    url=upload_file_data.get("url", "")
                )
            blocks.append(
                PromptBlock(
                    type=b_type, content=PromptBlockContent(upload_file=upload_file)
                )
            )

    return ClientMessage(
        type=d.get("type", "query"),
        project_id=d.get("project_id", ""),
        session_id=d.get("session_id", ""),
        local_msg_id=d.get("local_msg_id", ""),
        content=ClientMessageContent(query=QueryDetail(prompt=blocks)),
    ), d.get("session_id", "")


def _merge_tool_call_chunks(chunks: List[Any]) -> List[Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for chunk in chunks:
        # chunk can be dict or object
        if isinstance(chunk, dict):
            index = chunk.get("index")
            c_id = chunk.get("id")
            c_name = chunk.get("name")
            c_args = chunk.get("args")
        else:
            index = getattr(chunk, "index", None)
            c_id = getattr(chunk, "id", None)
            c_name = getattr(chunk, "name", None)
            c_args = getattr(chunk, "args", None)

        if index is None:
            continue

        # Normalize to string to avoid type errors during merge
        c_id_str = "".join(str(x) for x in c_id) if isinstance(c_id, list) else (c_id or "")
        c_name_str = "".join(str(x) for x in c_name) if isinstance(c_name, list) else (c_name or "")
        c_args_str = "".join(str(x) for x in c_args) if isinstance(c_args, list) else (c_args or "")

        if index not in merged:
            merged[index] = {
                "index": index,
                "id": c_id_str,
                "name": c_name_str,
                "args": c_args_str,
                "type": "tool_call",
            }
        else:
            merged[index]["id"] += c_id_str
            merged[index]["name"] += c_name_str
            merged[index]["args"] += c_args_str

    return list(merged.values())


def _item_to_server_messages(
        item: Dict[Any, Dict[str, Any]],
        *,
        session_id: str,
        query_msg_id: str,
        reply_id: str,
        sequence_id_start: int = 1,
        log_id: str = "",
) -> List[ServerMessage]:
    chunk, meta = item
    messages: List[ServerMessage] = []

    # Filter out messages from "tools" node to prevent internal model outputs from leaking as answers
    if (meta or {}).get("langgraph_node") == "tools":
        return messages

    def _make_message(
            msg_type: str, content: ServerMessageContent, finish: bool, seq: int
    ) -> ServerMessage:
        return ServerMessage(
            type=msg_type,
            session_id=session_id,
            query_msg_id=query_msg_id,
            reply_id=reply_id,
            msg_id=str(uuid.uuid4()),
            sequence_id=seq,
            finish=finish,
            content=content,
            log_id=log_id,
        )

    seq = sequence_id_start

    # Answer chunks (AIMessageChunk)
    if chunk.__class__.__name__ == "AIMessageChunk":
        text = getattr(chunk, "content", "")
        
        # Safely get finish_reason
        finish_reason = None
        try:
            resp_meta = getattr(chunk, "response_metadata", {})
            if resp_meta and isinstance(resp_meta, dict):
                finish_reason = resp_meta.get("finish_reason")
        except Exception:
            pass

        # Determine if this is the last chunk
        is_last_chunk = (meta or {}).get("chunk_position") == "last"
        is_finished = is_last_chunk or bool(finish_reason)

        # Check if this chunk involves tool calls
        has_tool_calls = bool(getattr(chunk, "tool_call_chunks", None)) or (finish_reason == "tool_calls")

        # Only emit answer if there is text content OR if it's a finish signal
        if text or (is_finished and not has_tool_calls):
            content = ServerMessageContent(answer=str(text) if text is not None else "")
            messages.append(
                _make_message(MESSAGE_TYPE_ANSWER, content, bool(is_finished), seq)
            )
            seq += 1

    # Final answer (AIMessage)
    if chunk.__class__.__name__ == "AIMessage":
        text = getattr(chunk, "content", "")
        if text:
            content = ServerMessageContent(answer=text)
            messages.append(_make_message(MESSAGE_TYPE_ANSWER, content, True, seq))
            seq += 1

    # Tool request from AIMessage (Complete)
    # Note: AIMessageChunk tool calls are handled in _iter_body_to_server_messages
    tool_calls = getattr(chunk, "tool_calls", None)
    if tool_calls and chunk.__class__.__name__ != "AIMessageChunk":
        items = tool_calls
        for tc in items:
            # Normalize parameters to dict
            raw_args = (
                tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            )
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    parameters = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    parameters = {}
            elif isinstance(raw_args, dict):
                parameters = raw_args
            else:
                parameters = {}
            tool_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            tool_name_str: str = str(tool_name or "")
            detail = ToolRequestDetail(
                tool_call_id=(
                                 tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                             )
                             or "",
                tool_name=tool_name_str,
                parameters={tool_name_str: parameters},
            )
            content = ServerMessageContent(tool_request=detail)
            messages.append(
                _make_message(MESSAGE_TYPE_TOOL_REQUEST, content, True, seq)
            )
            seq += 1

    return messages


def _iter_body_to_server_messages(
        items: Iterator[Dict[Any, Dict[str, Any]]],
        *,
        session_id: str,
        query_msg_id: str,
        reply_id: str,
        sequence_id_start: int = 1,
        log_id: str = "",
) -> Iterator[ServerMessage]:
    seq = sequence_id_start
    # Stable msg_id mapping per logical message stream
    # Keys are derived from meta to keep same msg_id across chunks
    stable_ids: Dict[Tuple[str, Any], str] = {}

    accumulated_tool_chunks: List[Any] = []
    accumulated_tool_response_content: Dict[str, str] = {}

    def _flush_tool_chunks(seq_num: int) -> Tuple[List[ServerMessage], int]:
        nonlocal accumulated_tool_chunks
        msgs: List[ServerMessage] = []
        if not accumulated_tool_chunks:
            return msgs, seq_num

        merged_tcs = _merge_tool_call_chunks(accumulated_tool_chunks)
        accumulated_tool_chunks = []
        for tc in merged_tcs:
            raw_args = tc.get("args", {})
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    parameters = parsed if isinstance(parsed, dict) else {}
                except Exception:
                    parameters = {}
            elif isinstance(raw_args, dict):
                parameters = raw_args
            else:
                parameters = {}
            tool_call_id = tc.get("id", "")
            tool_name = tc.get("name", "")

            detail = ToolRequestDetail(
                tool_call_id=tool_call_id or "",
                tool_name=tool_name or "",
                parameters={tool_name: parameters},
            )
            content = ServerMessageContent(tool_request=detail)
            msgs.append(
                ServerMessage(
                    type=MESSAGE_TYPE_TOOL_REQUEST,
                    session_id=session_id,
                    query_msg_id=query_msg_id,
                    reply_id=reply_id,
                    msg_id=str(uuid.uuid4()),
                    sequence_id=seq_num,
                    finish=True,
                    content=content,
                    log_id=log_id,
                )
            )
            seq_num += 1
        return msgs, seq_num

    for item in items:
        chunk, meta = item
        chunk_type = chunk.__class__.__name__
        is_last = (meta or {}).get("chunk_position") == "last"
        is_streaming = (meta or {}).get("chunk_position") is not None

        msgs_to_yield: List[ServerMessage] = []
        flushed_msgs: List[ServerMessage] = []

        # 0. Flush accumulated tool chunks if we receive something that is NOT an AIMessageChunk
        # OR if we receive an AIMessageChunk but it seems to be a new message (e.g. different id, though hard to track without state)
        # Simplest logic: If we have accumulated chunks, and we get a non-AIMessageChunk, flush.
        # Also, if we get an AIMessageChunk but it has no tool_call_chunks, it might be an answer chunk.
        # We need to be careful not to flush prematurely if the next chunk IS a continuation.

        # However, standard behavior: tool_call_chunks come in a contiguous sequence of AIMessageChunks.
        # If we see a ToolMessage, we definitely must flush.
        # If we see an AIMessageChunk with NO tool_call_chunks (e.g. text answer), we should probably flush too,
        # because usually tool calls and text content are either separate or tool calls come first.
        # But let's be safe: only flush on ToolMessage or if is_last=True on AIMessageChunk.

        if chunk_type == "ToolMessage" and accumulated_tool_chunks:
            f_msgs, seq = _flush_tool_chunks(seq)
            flushed_msgs.extend(f_msgs)

        # 1. Handle AIMessageChunk with tool_call_chunks (Streaming Tool Request)
        if chunk_type == "AIMessageChunk":
            tc_chunks = getattr(chunk, "tool_call_chunks", None)
            if tc_chunks:
                accumulated_tool_chunks.extend(tc_chunks)
            # If we have accumulated chunks but this chunk has NO tool_call_chunks,
            # it implies the tool definition phase is likely over.
            elif accumulated_tool_chunks:
                f_msgs, seq = _flush_tool_chunks(seq)
                flushed_msgs.extend(f_msgs)

            # Flush if this is the last chunk
            if is_last and accumulated_tool_chunks:
                f_msgs, seq = _flush_tool_chunks(seq)
                flushed_msgs.extend(f_msgs)

        # 2. Handle ToolMessage (Tool Response)
        elif chunk_type == "ToolMessage":
            tcid = getattr(chunk, "tool_call_id", "") or ""
            result = getattr(chunk, "content", "") or ""

            full_result = None
            should_emit = False

            if not is_streaming:
                full_result = result
                should_emit = True
            else:
                if tcid not in accumulated_tool_response_content:
                    accumulated_tool_response_content[tcid] = ""
                accumulated_tool_response_content[tcid] += str(result)

                if is_last:
                    full_result = accumulated_tool_response_content.pop(tcid)
                    should_emit = True

            if should_emit:
                detail = ToolResponseDetail(
                    tool_call_id=tcid,
                    code="0",
                    message="",
                    result=str(full_result),
                )
                content = ServerMessageContent(tool_response=detail)
                msgs_to_yield.append(
                    ServerMessage(
                        type=MESSAGE_TYPE_TOOL_RESPONSE,
                        session_id=session_id,
                        query_msg_id=query_msg_id,
                        reply_id=reply_id,
                        msg_id=str(uuid.uuid4()),
                        sequence_id=seq,
                        finish=True,
                        content=content,
                        log_id=log_id,
                    )
                )
                seq += 1

        # 3. Call _item_to_server_messages for everything else
        if chunk_type != "ToolMessage":
            inner_msgs = _item_to_server_messages(
                item,
                session_id=session_id,
                query_msg_id=query_msg_id,
                reply_id=reply_id,
                sequence_id_start=seq,
                log_id=log_id,
            )
            # Combine: flushed (previous) + inner (current)
            final_msgs = flushed_msgs + inner_msgs
            msgs_to_yield.extend(final_msgs)
            
            if inner_msgs:
                seq = inner_msgs[-1].sequence_id + 1
        else:
            # For ToolMessage, msgs_to_yield already contains the ToolResponse (from block 2).
            # We need to prepend flushed_msgs (from block 0).
            # Note: msgs_to_yield might contain ToolResponse or nothing (if not emitted yet).
            # But flushed_msgs MUST come before whatever is in msgs_to_yield.
            # However, msgs_to_yield is a list we are appending to.
            # So we should insert flushed_msgs at the beginning?
            # Or better, just construct a new list.
            
            # Current content of msgs_to_yield comes from block 2 (Tool Response).
            # flushed_msgs comes from block 0 (Tool Requests flushed).
            # Order: Tool Request -> Tool Response.
            final_msgs = flushed_msgs + msgs_to_yield
            msgs_to_yield = final_msgs


        for m in msgs_to_yield:
            # Derive a stable grouping base for this item
            group_base = (
                    meta.get("langgraph_checkpoint_ns")
                    or meta.get("checkpoint_ns")
                    or getattr(chunk, "id", None)
                    or meta.get("run_id")
                    or meta.get("langgraph_path")
                    or meta.get("langgraph_step")
            )

            # Derive grouping key per message type
            key: Tuple[str, Any]
            if m.type == MESSAGE_TYPE_TOOL_REQUEST and m.content.tool_request:
                tcid = m.content.tool_request.tool_call_id or None
                key = (MESSAGE_TYPE_TOOL_REQUEST, tcid or group_base)
            elif m.type == MESSAGE_TYPE_TOOL_RESPONSE and m.content.tool_response:
                tcid = m.content.tool_response.tool_call_id or None
                key = (MESSAGE_TYPE_TOOL_RESPONSE, tcid or group_base)
            elif m.type == MESSAGE_TYPE_ANSWER:
                # Prefer chunk.id to keep same msg_id across the entire answer stream
                key = (MESSAGE_TYPE_ANSWER, getattr(chunk, "id", None) or group_base)
            else:
                key = (m.type, group_base)

            if key not in stable_ids:
                stable_ids[key] = str(uuid.uuid4())
            m.msg_id = stable_ids[key]

            yield m


def iter_server_messages(
        items: Iterator[Dict[Any, Dict[str, Any]]],
        *,
        session_id: str,
        query_msg_id: str,
        local_msg_id: str,
        run_id: str,
        sequence_id_start: int = 1,
        log_id: str,
) -> Iterator[ServerMessage]:
    t0 = time.time()
    reply_id = str(uuid.uuid4())
    start_msg_id = str(uuid.uuid4())
    # message_start
    start_sm = ServerMessage(
        type=MESSAGE_TYPE_MESSAGE_START,
        session_id=session_id,
        query_msg_id=query_msg_id,
        reply_id=reply_id,
        msg_id=start_msg_id,
        sequence_id=sequence_id_start,
        finish=True,
        content=ServerMessageContent(
            message_start=MessageStartDetail(
                local_msg_id=local_msg_id, msg_id=query_msg_id, execute_id=run_id
            )
        ),
        log_id=log_id,
    )
    yield start_sm
    next_seq = sequence_id_start + 1
    last_seq = sequence_id_start
    try:
        # body stream
        for sm in _iter_body_to_server_messages(
                items,
                session_id=session_id,
                query_msg_id=query_msg_id,
                reply_id=reply_id,
                sequence_id_start=next_seq,
                log_id=log_id,
        ):
            yield sm
            last_seq = sm.sequence_id

        # message_end
        t_ms = int((time.time() - t0) * 1000)
        end_sm = ServerMessage(
            type=MESSAGE_TYPE_MESSAGE_END,
            session_id=session_id,
            query_msg_id=query_msg_id,
            reply_id=reply_id,
            msg_id=str(uuid.uuid4()),
            sequence_id=last_seq + 1,
            finish=True,
            content=ServerMessageContent(
                message_end=MessageEndDetail(
                    code=MESSAGE_END_CODE_SUCCESS,
                    message="",
                    token_cost=TokenCost(input_tokens=0, output_tokens=0, total_tokens=0),
                    time_cost_ms=t_ms,
                )
            ),
            log_id=log_id,
        )
        yield end_sm
    except Exception as ex:
        # message_end
        t_ms = int((time.time() - t0) * 1000)
        end_sm = ServerMessage(
            type=MESSAGE_TYPE_MESSAGE_END,
            session_id=session_id,
            query_msg_id=query_msg_id,
            reply_id=reply_id,
            msg_id=str(uuid.uuid4()),
            sequence_id=last_seq + 1,
            finish=True,
            content=ServerMessageContent(
                message_end=MessageEndDetail(
                    code="500",
                    message=str(ex),
                    time_cost_ms=t_ms,
                    token_cost=TokenCost(input_tokens=0, output_tokens=0, total_tokens=0),
                )
            ),
            log_id=log_id,
        )
        yield end_sm


def agent_iter_server_messages(
        items: Iterator[Dict[Any, Dict[str, Any]]],
        *,
        session_id: str,
        query_msg_id: str,
        local_msg_id: str,
        run_id: str,
        log_id: str,
) -> Iterator[ServerMessage]:
    return iter_server_messages(
        items,
        session_id=session_id,
        query_msg_id=query_msg_id,
        local_msg_id=local_msg_id,
        run_id=run_id,
        sequence_id_start=1,
        log_id=log_id,
    )
