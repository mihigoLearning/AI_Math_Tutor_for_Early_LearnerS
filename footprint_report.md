# footprint_report.md

Live size command output:

```bash
du -sh tutor/
80K    tutor/
```

## Per-component size table (tutor/)

| Component | Approx size |
|---|---:|
| `tutor/__init__.py` | < 1 KB |
| `tutor/engine.py` | ~ 8 KB |
| `tutor/demo.py` | ~ 6 KB |
| `tutor/curriculum_loader.py` | ~ 1 KB |
| `tutor/adaptive.py` | < 1 KB |
| `tutor/asr_adapt.py` | ~ 1 KB |
| `tutor/model.onnx` | ~ 1 KB |

Notes:
- Current tutor package size is far below the 75 MB limit.
- `tutor/model.onnx` is a real quantized artifact (int8 weights with in-graph dequantization).
