"""Distribution-aware active learning for Open-World Object Detection."""

from daowod.acquisition import compare_acquisition_strategies
from daowod.config import ExperimentConfig, load_config
from daowod.diagnostics import component_diagnostics, proposal_table, uncertainty_comparison
from daowod.experiment import (
    ActiveLearningCampaign,
    ExperimentResult,
    RoundResult,
    run_active_round,
)
from daowod.groups import ClassGroups
from daowod.prob_adapter import ProbAdapter, ProposalBatch
from daowod.scoring import (
    REQUIRED_STRATEGIES,
    STRATEGY_REGISTRY,
    StrategySpec,
    score_pool,
    select_images,
)

__all__ = [
    "REQUIRED_STRATEGIES",
    "STRATEGY_REGISTRY",
    "ActiveLearningCampaign",
    "ClassGroups",
    "ExperimentConfig",
    "ExperimentResult",
    "ProbAdapter",
    "ProposalBatch",
    "RoundResult",
    "StrategySpec",
    "compare_acquisition_strategies",
    "component_diagnostics",
    "load_config",
    "proposal_table",
    "run_active_round",
    "score_pool",
    "select_images",
    "uncertainty_comparison",
]
