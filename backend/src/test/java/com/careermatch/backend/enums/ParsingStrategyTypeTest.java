package com.careermatch.backend.enums;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ParsingStrategyTypeTest {

    @Test
    @DisplayName("Should have correct bean name for AI strategy")
    void AI_BeanName() {
        assertEquals("aiParsingStrategy", ParsingStrategyType.AI.getBeanName());
    }

    @Test
    @DisplayName("Should have correct bean name for ML strategy")
    void ML_BeanName() {
        assertEquals("mlParsingStrategy", ParsingStrategyType.ML.getBeanName());
    }

    @Test
    @DisplayName("Should have correct description for AI strategy")
    void AI_Description() {
        assertEquals("AI-based parsing using Gemini", ParsingStrategyType.AI.getDescription());
    }

    @Test
    @DisplayName("Should have correct description for ML strategy")
    void ML_Description() {
        assertEquals("ML-based parsing using Python ML service", ParsingStrategyType.ML.getDescription());
    }

    @Test
    @DisplayName("Should parse AI from uppercase string")
    void fromString_AI_Uppercase() {
        assertEquals(ParsingStrategyType.AI, ParsingStrategyType.fromString("AI"));
    }

    @Test
    @DisplayName("Should parse AI from lowercase string")
    void fromString_AI_Lowercase() {
        assertEquals(ParsingStrategyType.AI, ParsingStrategyType.fromString("ai"));
    }

    @Test
    @DisplayName("Should parse AI from mixed case string")
    void fromString_AI_MixedCase() {
        assertEquals(ParsingStrategyType.AI, ParsingStrategyType.fromString("Ai"));
    }

    @Test
    @DisplayName("Should parse ML from uppercase string")
    void fromString_ML_Uppercase() {
        assertEquals(ParsingStrategyType.ML, ParsingStrategyType.fromString("ML"));
    }

    @Test
    @DisplayName("Should parse ML from lowercase string")
    void fromString_ML_Lowercase() {
        assertEquals(ParsingStrategyType.ML, ParsingStrategyType.fromString("ml"));
    }

    @Test
    @DisplayName("Should parse ML from mixed case string")
    void fromString_ML_MixedCase() {
        assertEquals(ParsingStrategyType.ML, ParsingStrategyType.fromString("Ml"));
    }

    @Test
    @DisplayName("Should return null for null input")
    void fromString_Null() {
        assertNull(ParsingStrategyType.fromString(null));
    }

    @Test
    @DisplayName("Should return null for invalid string")
    void fromString_Invalid() {
        assertNull(ParsingStrategyType.fromString("INVALID"));
    }

    @Test
    @DisplayName("Should return null for empty string")
    void fromString_Empty() {
        assertNull(ParsingStrategyType.fromString(""));
    }

    @Test
    @DisplayName("Should return null for whitespace string")
    void fromString_Whitespace() {
        assertNull(ParsingStrategyType.fromString("   "));
    }

    @Test
    @DisplayName("Should have exactly 2 values")
    void values_Count() {
        assertEquals(2, ParsingStrategyType.values().length);
    }

    @Test
    @DisplayName("Should contain AI and ML values")
    void values_Contains() {
        ParsingStrategyType[] values = ParsingStrategyType.values();
        assertTrue(java.util.Arrays.asList(values).contains(ParsingStrategyType.AI));
        assertTrue(java.util.Arrays.asList(values).contains(ParsingStrategyType.ML));
    }

    @Test
    @DisplayName("Should return correct value from valueOf")
    void valueOf_AI() {
        assertEquals(ParsingStrategyType.AI, ParsingStrategyType.valueOf("AI"));
    }

    @Test
    @DisplayName("Should return correct value from valueOf for ML")
    void valueOf_ML() {
        assertEquals(ParsingStrategyType.ML, ParsingStrategyType.valueOf("ML"));
    }

    @Test
    @DisplayName("Should throw exception for invalid valueOf")
    void valueOf_Invalid() {
        assertThrows(IllegalArgumentException.class, () -> ParsingStrategyType.valueOf("INVALID"));
    }
}
