"""
Training configuration for TD Fish Follow.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Training schedule
    max_episodes: int = 2000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 50  # Episodes between evaluations
    eval_episodes: int = 10  # Number of episodes for evaluation
    
    # Checkpointing
    save_frequency: int = 100  # Episodes between model saves
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 200  # Episodes without improvement
    min_improvement: float = 0.01  # Minimum improvement threshold
    
    # Performance thresholds
    success_threshold: float = 15.0  # Average distance for "success"
    convergence_window: int = 100  # Episodes to average for convergence
    
    # Logging and monitoring
    log_frequency: int = 10  # Episodes between detailed logs
    tensorboard_logging: bool = True
    log_dir: str = "logs"
    
    # Visualization during training
    show_live_plots: bool = True
    plot_update_frequency: int = 5  # Episodes between plot updates
    save_training_videos: bool = False
    video_frequency: int = 100  # Episodes between video saves
    
    # Multi-run experiments
    num_seeds: int = 1  # Number of random seeds to run
    parallel_training: bool = False
    
    # Resource management
    device: str = "auto"  # "cpu", "cuda", or "auto"
    num_workers: int = 1  # For parallel data loading
    
    # Curriculum learning
    use_curriculum: bool = False
    curriculum_stages: Optional[list] = None
    stage_episodes: int = 200  # Episodes per curriculum stage
    
    # Target pattern scheduling
    pattern_schedule: Optional[dict] = None  # Pattern -> episode range
    random_pattern_prob: float = 0.1  # Probability of random pattern
    
    def __post_init__(self):
        """Set default curriculum and pattern schedule if not provided."""
        if self.use_curriculum and self.curriculum_stages is None:
            # Default curriculum: start easy, get harder
            self.curriculum_stages = [
                {'pattern': 'circular', 'target_speed': 30.0},
                {'pattern': 'figure8', 'target_speed': 40.0},
                {'pattern': 'zigzag', 'target_speed': 50.0},
                {'pattern': 'random_walk', 'target_speed': 50.0}
            ]
        
        if self.pattern_schedule is None:
            # Default: focus on random walk (most challenging)
            self.pattern_schedule = {
                'random_walk': (0, self.max_episodes),
                'circular': (0, self.max_episodes // 4),
                'figure8': (self.max_episodes // 4, self.max_episodes // 2)
            }
    
    def get_current_curriculum_stage(self, episode: int) -> dict:
        """Get current curriculum stage based on episode number."""
        if not self.use_curriculum:
            return None
        
        stage_idx = min(episode // self.stage_episodes, len(self.curriculum_stages) - 1)
        return self.curriculum_stages[stage_idx]
    
    def should_use_pattern(self, pattern: str, episode: int) -> bool:
        """Check if a pattern should be used at the current episode."""
        if pattern not in self.pattern_schedule:
            return False
        
        start_ep, end_ep = self.pattern_schedule[pattern]
        return start_ep <= episode < end_ep
    
    def get_available_patterns(self, episode: int) -> list:
        """Get list of patterns available at current episode."""
        available = []
        for pattern, (start_ep, end_ep) in self.pattern_schedule.items():
            if start_ep <= episode < end_ep:
                available.append(pattern)
        return available
    
    def should_save_checkpoint(self, episode: int) -> bool:
        """Check if checkpoint should be saved at current episode."""
        return episode % self.save_frequency == 0
    
    def should_evaluate(self, episode: int) -> bool:
        """Check if evaluation should be performed at current episode."""
        return episode % self.eval_frequency == 0
    
    def should_log(self, episode: int) -> bool:
        """Check if detailed logging should be performed."""
        return episode % self.log_frequency == 0
    
    def should_update_plots(self, episode: int) -> bool:
        """Check if plots should be updated."""
        return self.show_live_plots and episode % self.plot_update_frequency == 0
    
    def should_save_video(self, episode: int) -> bool:
        """Check if training video should be saved."""
        return self.save_training_videos and episode % self.video_frequency == 0
