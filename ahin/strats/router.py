from typing import Dict, Any, List, Optional, Tuple
from ahin.core import ResponseStrategyProtocol


class RouterStrategy:
    """
    Router/Chain strategy that passes input through a list of other strategies
    and stops at the first one that returns True for a match.
    
    This allows composing multiple strategies together, with each strategy
    having a chance to handle the input. The first strategy that successfully
    matches (returns True) wins.
    
    Example usage:
        strategy = RouterStrategy(config, [
            ConversationalStrategy(config),
            SemanticStrategy(config),
            LLMStrategy(config),
            FallbackStrategy(config)  # Always matches, provides defaults
        ])
    """
    
    def __init__(self, config: Dict[str, Any], strategies: Optional[List[ResponseStrategyProtocol]] = None):
        """
        Initialize the router with a config and list of strategies.
        
        Args:
            config: Configuration dictionary
            strategies: List of strategy instances to route through.
                        They will be tried in order until one returns True (matched).
        """
        self.config = config
        self.strategies = strategies if strategies is not None else []
    
    def add_strategy(self, strategy: ResponseStrategyProtocol) -> None:
        """
        Add a strategy to the routing chain.
        
        Args:
            strategy: Strategy instance to add
        """
        self.strategies.append(strategy)
    
    def remove_strategy(self, strategy: ResponseStrategyProtocol) -> None:
        """
        Remove a strategy from the routing chain.
        
        Args:
            strategy: Strategy instance to remove
        """
        if strategy in self.strategies:
            self.strategies.remove(strategy)
    
    def clear_strategies(self) -> None:
        """Clear all strategies from the routing chain."""
        self.strategies.clear()
    
    def generate_response(self, text: str) -> Tuple[bool, str]:
        """
        Generate a response by trying each strategy in order.
        Returns the first response where the strategy indicates a match (returns True).
        
        Args:
            text: Input text to generate a response for
            
        Returns:
            Tuple of (matched: bool, response: str)
            - matched: True if any strategy matched
            - response: Response from the first matching strategy, or empty string
        """
        if not text:
            return (False, "")
        
        if not self.strategies:
            # No strategies configured
            return (False, "No strategies configured.")
        
        for strategy in self.strategies:
            try:
                matched, response = strategy.generate_response(text)
                
                # If this strategy matched, return its response
                if matched:
                    return (True, response)
                    
            except Exception as e:
                # Log error but continue to next strategy
                print(f"Strategy {strategy.__class__.__name__} failed: {e}")
                continue
        
        # If we get here, no strategy matched
        return (False, "")
