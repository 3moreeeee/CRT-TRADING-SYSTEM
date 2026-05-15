from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd

from .config import CRTConfig
from .data import build_timeframes, validate_ohlcv
from .detector import detect_crt_setups
from .features import build_feature_frame
from .labels import build_label_frame
from .modeling import CRTModelBundle, train_crt_model
from .types import CRTSetup, ExecutionPlan


class CRTPipeline:
    def __init__(self, config: CRTConfig | None = None):
        self.config = config or CRTConfig()

    def build_market_frames(self, base_dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        frames = build_timeframes(base_dataframe, self.config.setup_timeframe, self.config.execution_timeframe)
        return frames.setup, frames.execution

    def detect_setups(self, setup_frame: pd.DataFrame, symbol: str = "MARKET") -> List[CRTSetup]:
        return detect_crt_setups(setup_frame, symbol, self.config.setup_timeframe, self.config.execution_timeframe, config=self.config)

    def build_execution_plans(self, setups: List[CRTSetup], execution_frame: pd.DataFrame) -> List[ExecutionPlan]:
        plans: List[ExecutionPlan] = []
        from .execution import build_execution_plan

        for setup in setups:
            plans.append(build_execution_plan(setup, execution_frame, self.config))
        return plans

    def build_dataset(self, base_dataframe: pd.DataFrame, symbol: str = "MARKET") -> pd.DataFrame:
        setup_frame, execution_frame = self.build_market_frames(base_dataframe)
        setups = self.detect_setups(setup_frame, symbol=symbol)
        plans = self.build_execution_plans(setups, execution_frame)
        feature_frame = build_feature_frame(setups, plans, setup_frame, execution_frame, config=self.config)
        label_frame = build_label_frame(setups, plans, execution_frame, config=self.config, target_mode=self.config.target_mode)
        if feature_frame.empty or label_frame.empty:
            return pd.DataFrame()
        dataset = feature_frame.merge(label_frame, on="setup_id", how="inner", suffixes=("", "_label"))
        dataset["target"] = dataset["label"].astype(int)
        return dataset

    def train(self, base_dataframe: pd.DataFrame, symbol: str = "MARKET") -> CRTModelBundle:
        dataset = self.build_dataset(base_dataframe, symbol=symbol)
        if dataset.empty:
            raise ValueError("No CRT samples were produced from the supplied market data")
        return train_crt_model(dataset, target_column="target")
