#!/usr/bin/env python3
"""
Hybrid Multi-Agent Curiosity Arena - Main Entry Point

A groundbreaking RL system combining PPO + Curiosity with A3C Competitive learning
in a unified multi-agent environment.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.base_config import HybridArenaConfig, load_config, AgentType
from web.unified_interface import create_app


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hybrid_arena.log')
        ]
    )


def print_banner():
    """Print startup banner"""
    banner = """
🌊 ═══════════════════════════════════════════════════════════════════════════════ 🌊
   
   🧠 HYBRID MULTI-AGENT CURIOSITY ARENA 🧠
   
   A Revolutionary Reinforcement Learning Environment
   Combining PPO + Curiosity with A3C Competitive Learning
   
🌊 ═══════════════════════════════════════════════════════════════════════════════ 🌊

🎯 AGENT TYPES:
   🔍 Curious Agents    - PPO + Intrinsic Curiosity Module (exploration-focused)
   ⚔️  Competitive Agents - A3C with trust regions (resource competition)
   🔄 Hybrid Agents     - Dynamic strategy switching
   🧬 Adaptive Agents   - Meta-learning optimal strategies

🌍 ENVIRONMENT:
   📊 152D observation space (30 sensor rays + proprioception)
   🎮 Real-time multi-agent interactions
   📈 Advanced analytics and visualization
   🔬 Research-grade experimental framework

🚀 INNOVATION:
   • First-of-its-kind algorithm combination
   • Real-time strategy switching during training
   • Population-level learning dynamics
   • Cross-paradigm agent interactions
"""
    print(banner)


def print_config_summary(config: HybridArenaConfig):
    """Print configuration summary"""
    print("📋 CONFIGURATION SUMMARY:")
    print("=" * 50)
    print(f"🌍 Environment: {config.environment.world_width}x{config.environment.world_height}")
    print(f"👥 Total Agents: {config.training.total_agents}")
    
    print("\n🤖 Agent Composition:")
    for agent_type, count in config.training.agent_composition.items():
        emoji_map = {
            AgentType.CURIOUS: "🔍",
            AgentType.COMPETITIVE: "⚔️",
            AgentType.HYBRID: "🔄", 
            AgentType.ADAPTIVE: "🧬"
        }
        emoji = emoji_map.get(agent_type, "🤖")
        print(f"   {emoji} {agent_type.value.title()}: {count}")
    
    print(f"\n🌐 Web Interface: http://localhost:{config.web.port}")
    print(f"🎯 Device: {config.device}")
    print(f"📝 Log Level: {config.log_level}")
    print("=" * 50)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hybrid Multi-Agent Curiosity Arena",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run with default configuration
  python main.py --config custom.yaml     # Run with custom configuration
  python main.py --port 8000              # Run on different port
  python main.py --agents curious:4 competitive:4  # Custom agent composition
  python main.py --headless               # Run without web interface
  python main.py --experiment mixed_pop   # Run predefined experiment
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration YAML file'
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        help='Web interface port (default: 7000)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Web interface host (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--agents',
        nargs='+',
        help='Agent composition (e.g., curious:2 competitive:3 hybrid:1)'
    )
    
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run without web interface (training only)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['pure_curious', 'pure_competitive', 'mixed_pop', 'hybrid_evolution'],
        help='Run predefined experiment'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        default='auto',
        help='PyTorch device (default: auto)'
    )
    
    return parser.parse_args()


def apply_cli_overrides(config: HybridArenaConfig, args):
    """Apply command line argument overrides to configuration"""
    
    # Port override
    if args.port:
        config.web.port = args.port
    
    # Host override
    if args.host:
        config.web.host = args.host
    
    # Agent composition override
    if args.agents:
        new_composition = {}
        total_agents = 0
        
        for agent_spec in args.agents:
            if ':' not in agent_spec:
                print(f"Warning: Invalid agent specification '{agent_spec}'. Use format 'type:count'")
                continue
            
            agent_type_str, count_str = agent_spec.split(':', 1)
            try:
                count = int(count_str)
                agent_type = AgentType(agent_type_str.lower())
                new_composition[agent_type] = count
                total_agents += count
            except (ValueError, KeyError):
                print(f"Warning: Invalid agent specification '{agent_spec}'")
                continue
        
        if new_composition:
            config.training.agent_composition = new_composition
            config.training.total_agents = total_agents
    
    # Experiment presets
    if args.experiment:
        if args.experiment == 'pure_curious':
            config.training.agent_composition = {AgentType.CURIOUS: 8}
            config.training.total_agents = 8
        elif args.experiment == 'pure_competitive':
            config.training.agent_composition = {AgentType.COMPETITIVE: 8}
            config.training.total_agents = 8
        elif args.experiment == 'mixed_pop':
            config.training.agent_composition = {
                AgentType.CURIOUS: 2,
                AgentType.COMPETITIVE: 2,
                AgentType.HYBRID: 2,
                AgentType.ADAPTIVE: 2
            }
            config.training.total_agents = 8
        elif args.experiment == 'hybrid_evolution':
            config.training.agent_composition = {
                AgentType.HYBRID: 4,
                AgentType.ADAPTIVE: 4
            }
            config.training.total_agents = 8
    
    # Other overrides
    if args.seed:
        config.seed = args.seed
    
    if args.device:
        config.device = args.device
    
    if args.log_level:
        config.log_level = args.log_level


def run_headless_training(config: HybridArenaConfig):
    """Run headless training without web interface"""
    print("🚀 Starting headless training...")
    
    # Import training components
    from training.hybrid_trainer import HybridTrainer
    from environment.unified_waterworld import UnifiedWaterworld
    
    # Create environment and trainer
    env = UnifiedWaterworld(config)
    trainer = HybridTrainer(config, env)
    
    # Run training
    trainer.train()
    
    print("✅ Training completed!")


def run_web_interface(config: HybridArenaConfig):
    """Run web interface"""
    print("🌐 Starting web interface...")
    
    # Create Flask app
    app, socketio = create_app(config)
    
    print(f"🎮 Open your browser to: http://localhost:{config.web.port}")
    print("\n🎯 CONTROLS:")
    print("   • Real-time agent population adjustment")
    print("   • Live parameter tuning during training")
    print("   • Strategy switching for individual agents")
    print("   • Comparative performance analytics")
    print("   • Experiment designer and configuration export")
    print("\n⚡ FEATURES:")
    print("   • Multi-paradigm learning visualization")
    print("   • Population dynamics analysis")
    print("   • Strategy emergence detection")
    print("   • Cross-agent interaction patterns")
    print("\n🔬 RESEARCH APPLICATIONS:")
    print("   • Strategy dominance analysis")
    print("   • Emergent cooperation studies")
    print("   • Adaptive learning research")
    print("   • Resource efficiency optimization")
    
    try:
        # Run the web server
        socketio.run(
            app,
            host=config.web.host,
            port=config.web.port,
            debug=config.web.debug
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down web interface...")
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        print("💡 Make sure the port is not in use and dependencies are installed")


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Print banner
    print_banner()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply CLI overrides
        apply_cli_overrides(config, args)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Set random seed if specified
        if config.seed is not None:
            import torch
            import numpy as np
            import random
            
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
            print(f"🎲 Random seed set to: {config.seed}")
        
        # Run appropriate mode
        if args.headless:
            run_headless_training(config)
        else:
            run_web_interface(config)
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("💡 Check the logs for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()
