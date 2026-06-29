"""
VisionLanguageGame — Thin shapiq.Game adapter for HuggingFace VLMs.

All masking, batching, and model-forward logic is delegated to an
ImageImputer. The Game only handles Shapley scheduling concerns.

Usage:
    factory = ImageImputerFactory()
    imputer = factory.build(model, processor, input_image, input_text)
    game = VisionLanguageGame(imputer, batch_size=64)
"""

import numpy as np
from shapiq import Game


class VisionLanguageGame(Game):
    """
    A shapiq.Game adapter for HuggingFace CLIP / SigLIP.

    Imputer-based only. For the original built-in masking path,
    see src.game_huggingface.VisionLanguageGame.
    """

    def __init__(
        self,
        imputer,       # ImageImputer (required)
        batch_size: int = 64,
        verbose: bool = False,
    ):
        self._imputer = imputer
        self._batch_size = batch_size

        # Player counts come from the imputer's layout
        self.n_players_image = imputer.n_players_image
        self.n_players_text = imputer.n_players_text

        # Normalization values
        coalitions = np.zeros((2, self.n_players_image + self.n_players_text), dtype=bool)
        coalitions[1, :] = True
        game_output = self.value_function(coalitions=coalitions)
        self.empty_value = float(game_output[0])
        self.full_value = float(game_output[1])

        if verbose:
            print(f"Similarity of the Image and Text: {self.full_value} "
                  f"(empty_value={self.empty_value})")

        super().__init__(
            n_players=self.n_players_image + self.n_players_text,
            normalize=True,
            normalization_value=self.empty_value,
        )

    # ─── Delegated properties (backward compat with notebooks) ──────────

    @property
    def inputs(self):
        """Raw HuggingFace processor output (BatchEncoding)."""
        return self._imputer.inputs_raw

    @property
    def processor(self):
        """HuggingFace processor for image/text preprocessing."""
        return self._imputer.processor

    def cleanup(self) -> None:
        """Release any persistent model state held by the imputer's masker.

        Must be called when this game is discarded if the model is reused
        across games (e.g. AttentionMasker leaves forward hooks on the model).
        """
        self._imputer.cleanup()

    # ─── Value functions (delegate to Imputer) ───────────────────────────

    def value_function(self, coalitions, batch_size=None):
        """
        Input: np.ndarray[bool] shape (n_coalitions, n_players).
        Output: np.ndarray shape (n_coalitions,). Diagonal similarity scores.
        """
        if batch_size is None:
            batch_size = self._batch_size
        return self._imputer.forward_1d(coalitions, batch_size=batch_size)

    def value_function_crossmodal(self, coalitions_image, coalitions_text, batch_size=None):
        """
        Input: np.ndarray[bool] shapes (N_img, n_players_image) and (N_txt, n_players_text).
        Output: np.ndarray shape (N_img, N_txt). Full interaction matrix.
        """
        if batch_size is None:
            batch_size = self._batch_size
        return self._imputer.forward_crossmodal(
            coalitions_image, coalitions_text, batch_size=batch_size,
        )