package com.careermatch.backend.parse;

import com.careermatch.backend.enums.ParsingStrategyType;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.Map;

/**
 * Factory to get the appropriate parsing strategy
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class ParsingStrategyFactory {
    
    private final Map<String, ResumeParsingStrategy> strategies;
    
    @Value("${resume.parsing.strategy:AI}")
    private String defaultStrategyValue;
    
    /**
     * Get parsing strategy by enum type
     * @param strategyType type of strategy
     * @return the appropriate parsing strategy
     */
    public ResumeParsingStrategy getStrategy(ParsingStrategyType strategyType) {
        if (strategyType == null) {
            return getDefaultStrategy();
        }
        
        ResumeParsingStrategy strategy = strategies.get(strategyType.getBeanName());
        
        if (strategy == null) {
            log.error("Strategy bean not found: {}", strategyType.getBeanName());
            throw new IllegalStateException("Parsing strategy not available: " + strategyType);
        }
        
        log.info("Selected parsing strategy: {} - {}", 
            strategyType, strategyType.getDescription());
        return strategy;
    }
    
    /**
     * Get parsing strategy by string name
     * @param strategyName name of strategy (AI or ML)
     * @return the appropriate parsing strategy
     */
    public ResumeParsingStrategy getStrategy(String strategyName) {
        ParsingStrategyType strategyType = ParsingStrategyType.fromString(strategyName);
        
        if (strategyType == null) {
            log.warn("Unknown strategy: {}, falling back to default", strategyName);
            return getDefaultStrategy();
        }
        
        return getStrategy(strategyType);
    }
    
    /**
     * Get default parsing strategy
     * @return default strategy
     */
    public ResumeParsingStrategy getDefaultStrategy() {
        ParsingStrategyType defaultType = ParsingStrategyType.fromString(defaultStrategyValue);
        
        if (defaultType == null) {
            log.warn("Invalid default strategy configured: {}, using AI", defaultStrategyValue);
            defaultType = ParsingStrategyType.AI;
        }
        
        return getStrategy(defaultType);
    }
    
    /**
     * Get default strategy type
     * @return default ParsingStrategyType
     */
    public ParsingStrategyType getDefaultStrategyType() {
        ParsingStrategyType defaultType = ParsingStrategyType.fromString(defaultStrategyValue);
        return defaultType != null ? defaultType : ParsingStrategyType.AI;
    }
}