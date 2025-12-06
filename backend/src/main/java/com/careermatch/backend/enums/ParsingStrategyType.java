package com.careermatch.backend.enums;

/**
 * Enum representing different resume parsing strategies
 */
public enum ParsingStrategyType {
    /**
     * AI-based parsing using Gemini
     */
    AI("aiParsingStrategy", "AI-based parsing using Gemini"),
    
    /**
     * ML-based parsing using Python ML service
     */
    ML("mlParsingStrategy", "ML-based parsing using Python ML service");
    
    private final String beanName;
    private final String description;
    
    ParsingStrategyType(String beanName, String description) {
        this.beanName = beanName;
        this.description = description;
    }
    
    public String getBeanName() {
        return beanName;
    }
    
    public String getDescription() {
        return description;
    }
    
    /**
     * Get enum from string value (case-insensitive)
     * @param value string value
     * @return ParsingStrategyType or null if not found
     */
    public static ParsingStrategyType fromString(String value) {
        if (value == null) {
            return null;
        }
        
        try {
            return ParsingStrategyType.valueOf(value.toUpperCase());
        } catch (IllegalArgumentException e) {
            return null;
        }
    }
}