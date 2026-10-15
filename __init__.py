"""ComfyUI V3 entry point; load the host-dependent engine when requested."""
WEB_DIRECTORY = "./web"

__all__ = ["comfy_entrypoint"]


async def comfy_entrypoint():
    from .drama3_t8.nodes import comfy_entrypoint as load_extension
    return await load_extension()
