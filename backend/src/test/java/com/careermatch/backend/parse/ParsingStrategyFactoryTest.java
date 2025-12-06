package com.careermatch.backend.parse;

import com.careermatch.backend.enums.ParsingStrategyType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class ParsingStrategyFactoryTest {

    @Mock
    private ResumeParsingStrategy aiStrategy;

    @Mock
    private ResumeParsingStrategy mlStrategy;

    private ParsingStrategyFactory parsingStrategyFactory;

    @BeforeEach
    void setUp() {
        Map<String, ResumeParsingStrategy> strategies = new HashMap<>();
        strategies.put("aiParsingStrategy", aiStrategy);
        strategies.put("mlParsingStrategy", mlStrategy);

        parsingStrategyFactory = new ParsingStrategyFactory(strategies);
        ReflectionTestUtils.setField(parsingStrategyFactory, "defaultStrategyValue", "AI");
    }

    @Test
    @DisplayName("Should return AI strategy when AI type is requested")
    void getStrategy_AIType() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy(ParsingStrategyType.AI);

        assertNotNull(result);
        assertEquals(aiStrategy, result);
    }

    @Test
    @DisplayName("Should return ML strategy when ML type is requested")
    void getStrategy_MLType() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy(ParsingStrategyType.ML);

        assertNotNull(result);
        assertEquals(mlStrategy, result);
    }

    @Test
    @DisplayName("Should return default strategy when null type is requested")
    void getStrategy_NullType() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy((ParsingStrategyType) null);

        assertNotNull(result);
        assertEquals(aiStrategy, result);
    }

    @Test
    @DisplayName("Should return AI strategy for 'AI' string")
    void getStrategy_AIString() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy("AI");

        assertNotNull(result);
        assertEquals(aiStrategy, result);
    }

    @Test
    @DisplayName("Should return ML strategy for 'ML' string")
    void getStrategy_MLString() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy("ML");

        assertNotNull(result);
        assertEquals(mlStrategy, result);
    }

    @Test
    @DisplayName("Should return AI strategy for lowercase 'ai' string")
    void getStrategy_LowercaseAIString() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy("ai");

        assertNotNull(result);
        assertEquals(aiStrategy, result);
    }

    @Test
    @DisplayName("Should return ML strategy for lowercase 'ml' string")
    void getStrategy_LowercaseMLString() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy("ml");

        assertNotNull(result);
        assertEquals(mlStrategy, result);
    }

    @Test
    @DisplayName("Should return default strategy for unknown string")
    void getStrategy_UnknownString() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy("UNKNOWN");

        assertNotNull(result);
        assertEquals(aiStrategy, result);
    }

    @Test
    @DisplayName("Should return default strategy for null string")
    void getStrategy_NullString() {
        ResumeParsingStrategy result = parsingStrategyFactory.getStrategy((String) null);

        assertNotNull(result);
        assertEquals(aiStrategy, result);
    }

    @Test
    @DisplayName("Should throw exception when strategy bean not found")
    void getStrategy_BeanNotFound() {
        Map<String, ResumeParsingStrategy> emptyStrategies = new HashMap<>();
        ParsingStrategyFactory factoryWithNoStrategies = new ParsingStrategyFactory(emptyStrategies);

        assertThrows(IllegalStateException.class,
                () -> factoryWithNoStrategies.getStrategy(ParsingStrategyType.AI));
    }

    @Test
    @DisplayName("Should return default strategy type")
    void getDefaultStrategyType() {
        ParsingStrategyType result = parsingStrategyFactory.getDefaultStrategyType();

        assertEquals(ParsingStrategyType.AI, result);
    }

    @Test
    @DisplayName("Should return AI when default strategy value is invalid")
    void getDefaultStrategyType_InvalidDefault() {
        ReflectionTestUtils.setField(parsingStrategyFactory, "defaultStrategyValue", "INVALID");

        ParsingStrategyType result = parsingStrategyFactory.getDefaultStrategyType();

        assertEquals(ParsingStrategyType.AI, result);
    }

    @Test
    @DisplayName("Should return ML default strategy when configured")
    void getDefaultStrategy_MLConfigured() {
        ReflectionTestUtils.setField(parsingStrategyFactory, "defaultStrategyValue", "ML");

        ResumeParsingStrategy result = parsingStrategyFactory.getDefaultStrategy();

        assertEquals(mlStrategy, result);
    }

    @Test
    @DisplayName("Should return ML strategy type when configured as default")
    void getDefaultStrategyType_MLConfigured() {
        ReflectionTestUtils.setField(parsingStrategyFactory, "defaultStrategyValue", "ML");

        ParsingStrategyType result = parsingStrategyFactory.getDefaultStrategyType();

        assertEquals(ParsingStrategyType.ML, result);
    }

    @Test
    @DisplayName("Should fallback to AI when invalid default configured")
    void getDefaultStrategy_InvalidDefault() {
        ReflectionTestUtils.setField(parsingStrategyFactory, "defaultStrategyValue", "INVALID");

        ResumeParsingStrategy result = parsingStrategyFactory.getDefaultStrategy();

        assertEquals(aiStrategy, result);
    }
}
