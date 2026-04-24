"""Root demo entrypoint required by the assignment brief.

Runs the child-facing Gradio app from `tutor/demo.py`.
"""

from tutor.demo import build_ui


if __name__ == "__main__":  # pragma: no cover
    build_ui().launch()
