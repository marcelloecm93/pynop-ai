"""Giskard adapter — bridges SafetyPipeline to Giskard's Model interface."""

import asyncio

import numpy as np
import pandas as pd
from giskard import Model

from pynop.exceptions import GuardRejection


def create_pipeline_model(pipeline) -> Model:
    """Wrap a SafetyPipeline as a Giskard Model for use with giskard.scan()."""

    def _predict(df: pd.DataFrame) -> np.ndarray:
        results = []
        for _, row in df.iterrows():
            prompt = row["question"]
            try:
                result = asyncio.run(pipeline.run(prompt))
                results.append(result.output)
            except GuardRejection as e:
                results.append(f"[REJECTED] {e}")
            except Exception as e:
                results.append(f"[ERROR] {e}")
        return np.array(results)

    return Model(
        model=_predict,
        model_type="text_generation",
        name="pynop-SafetyPipeline",
        description="LLM safety pipeline with input/output guards",
        feature_names=["question"],
    )
