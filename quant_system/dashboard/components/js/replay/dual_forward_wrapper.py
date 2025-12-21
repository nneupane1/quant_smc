"""
dual_forward_wrapper.py
Wrap two ForwardEngine instances:
 • forward_A (model version A)
 • forward_B (model version B)
Ensures identical bar input, synchronized timing, and separate equity/risk states.
"""

class DualForwardWrapper:

    def __init__(self, forward_A, forward_B):
        self.A = forward_A
        self.B = forward_B

    def on_bar(self, asset, bar_15m):
        """Feed exact same bar to both engines."""
        self.A.on_bar(asset, bar_15m)
        self.B.on_bar(asset, bar_15m)

    def reset(self):
        self.A.reset()
        self.B.reset()
