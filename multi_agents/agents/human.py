import json


class HumanAgent:
    def __init__(self, websocket=None, stream_output=None, headers=None):
        self.websocket = websocket
        self.stream_output = stream_output
        self.headers = headers or {}

    async def review_plan(self, research_state: dict):
        print(f"HumanAgent websocket: {self.websocket}")
        print(f"HumanAgent stream_output: {self.stream_output}")
        task = research_state.get("task")
        layout = research_state.get("sections")

        user_feedback = None

        if task.get("include_human_feedback"):
            # 如果提供了 websocket（如来自 web 应用），把响应流式发给用户
            if self.websocket and self.stream_output:
                try:
                    await self.stream_output(
                        "human_feedback",
                        "request",
                        f"Any feedback on this plan of topics to research? {layout}? If not, please reply with 'no'.",
                        self.websocket,
                    )
                    # 因为 websocket 在 websocket_manager 中被包装进 CustomLogsHandler
                    socket = self.websocket.websocket
                    # 服务器的接收循环拥有 socket 并把反馈转发到这里。
                    queue = getattr(getattr(socket, "state", None), "human_feedback", None)
                    response = await (queue.get() if queue is not None else socket.receive_text())
                    print(f"Received response: {response}", flush=True)
                    response_data = json.loads(response)
                    if response_data.get("type") == "human_feedback":
                        user_feedback = response_data.get("content")
                    else:
                        print(
                            f"Unexpected response type: {response_data.get('type')}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"Error receiving human feedback: {e}", flush=True)
            # 否则在控制台提示用户反馈
            else:
                user_feedback = input(
                    f"Any feedback on this plan? {layout}? If not, please reply with 'no'.\n>> "
                )

        from .utils.none_sentinels import is_human_plan_approval

        if not user_feedback or is_human_plan_approval(user_feedback):
            user_feedback = None

        plan_revision_count = research_state.get("plan_revision_count", 0)
        if user_feedback:
            plan_revision_count += 1

        print(f"User feedback before return: {user_feedback}")

        return {
            "human_feedback": user_feedback,
            "plan_revision_count": plan_revision_count,
        }
