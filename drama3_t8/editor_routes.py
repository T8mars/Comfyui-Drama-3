"""ComfyUI's own local HTTP transport for CPU edit planning, no model bridge."""
import json

from aiohttp import web

from .editor import plan


def register_routes():
    from server import PromptServer
    server=PromptServer.instance
    if getattr(server,'_drama3_t8_editor_routes',False):
        return
    @server.routes.post('/drama3_t8/editor/plan')
    async def editor_plan(request):
        if request.content_length is not None and request.content_length>28_000_000:
            raise web.HTTPRequestEntityTooLarge(max_size=28_000_000,actual_size=request.content_length)
        try:
            raw=await request.text()
            if len(raw)>28_000_000:
                raise ValueError('Editor request is too large')
            payload=json.loads(raw)
            if not isinstance(payload,dict):
                raise ValueError('Editor request must be an object')
            return web.json_response(plan(payload.get('analysis'),payload.get('edits',{})))
        except (ValueError,TypeError,KeyError,OverflowError,RecursionError) as exc:
            return web.json_response({'error':str(exc)},status=400)
    server._drama3_t8_editor_routes=True
