"""
Console summary of a registered model, used by the `__main__` block of every model file:

    python -m models.fcos_detector
    python -m models.archive.stage5_unified_detector

or for any number of models at once (no args = every registered model):

    python -m models.summary fcos archive.stage4

For each size preset it prints the backbone widths and depths, the model's own settings
(classes, anchors, slots, ...), the parameter count with its fp32 size, and the output shapes
for a dummy batch of the model's INPUT_SHAPE.
"""
import torch

# Model attributes worth showing when a class defines them.
_SETTINGS = ("num_classes", "num_anchors", "anchors_per_scale", "max_objects", "fpn_channels", "grid_sizes")


def _shapes(output):
    """Shapes of a tensor, or of a (nested) dict of tensors keyed by grid size."""
    if isinstance(output, dict):
        return {key: _shapes(value) for key, value in output.items()}
    return tuple(output.shape)


def print_summary(name: str, batch_size: int = 2, **overrides) -> None:
    """Print one block per size preset of the model registered as *name* (see models.MODEL_NAMES)."""
    from models import _REGISTRY, build_model  # imported here: model files import this module lazily

    cls, presets = _REGISTRY[name]
    channels_in, height, width = cls.INPUT_SHAPE
    print("=" * 88)
    print(f"{cls.__name__}   build_model({name!r})   [{cls.__module__}]")
    doc = (cls.__doc__ or "").strip().splitlines()
    if doc:
        print(f"  {doc[0].strip()}")
    print(f"  input: ({batch_size}, {channels_in}, {height}, {width})")
    print("=" * 88)

    for size in (list(presets) if presets else [None]):
        model = build_model(name, size=size or "s", **overrides).eval()
        total = sum(p.numel() for p in model.parameters())
        with torch.no_grad():
            output = model(torch.zeros(batch_size, channels_in, height, width))

        print(f"[{size or '-'}]  params {total:>12,}   ({total * 4 / 2**20:6.2f} MB fp32)")
        if size:
            preset = presets[size]
            print(f"      channels {list(preset.channels)}   blocks {list(preset.blocks)}")
        settings = {key: getattr(model, key) for key in _SETTINGS if hasattr(model, key)}
        if settings:
            print("      " + "   ".join(f"{key} {value}" for key, value in settings.items()))
        print(f"      output {_shapes(output)}")
    print()


if __name__ == "__main__":
    import sys

    from models import MODEL_NAMES

    for model_name in sys.argv[1:] or MODEL_NAMES:
        print_summary(model_name)
