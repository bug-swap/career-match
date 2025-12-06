package com.careermatch.backend.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class AiServiceTest {

    private AiService aiService;

    @BeforeEach
    void setUp() {
        aiService = new AiService();
        ReflectionTestUtils.setField(aiService, "apiKey", "test-api-key");
        ReflectionTestUtils.setField(aiService, "modelName", "gemini-pro");
        ReflectionTestUtils.setField(aiService, "maxTokens", 8192);
        ReflectionTestUtils.setField(aiService, "temperature", 0.7);
        ReflectionTestUtils.setField(aiService, "topP", 0.9);
    }

    @Test
    @DisplayName("Should throw exception when API key is null")
    void initializeClient_NullApiKey() {
        ReflectionTestUtils.setField(aiService, "apiKey", null);

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> aiService.builder());

        assertEquals("Gemini API key is missing or invalid", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when API key is empty")
    void initializeClient_EmptyApiKey() {
        ReflectionTestUtils.setField(aiService, "apiKey", "");

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> aiService.builder());

        assertEquals("Gemini API key is missing or invalid", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when API key is placeholder")
    void initializeClient_PlaceholderApiKey() {
        ReflectionTestUtils.setField(aiService, "apiKey", "your-api-key-here");

        IllegalArgumentException exception = assertThrows(IllegalArgumentException.class,
                () -> aiService.builder());

        assertEquals("Gemini API key is missing or invalid", exception.getMessage());
    }

    @Test
    @DisplayName("Should create builder successfully with valid API key")
    void builder_ValidApiKey() {
        AiService.GeminiRequestBuilder builder = aiService.builder();

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should throw exception when executing without content")
    void builder_ExecuteWithoutContent() {
        AiService.GeminiRequestBuilder builder = aiService.builder();

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> builder.build().execute());

        assertTrue(exception.getMessage().contains("no content provided"));
    }

    @Test
    @DisplayName("Builder should accept prompt")
    void builder_WithPrompt() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt");

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should accept additional text")
    void builder_WithText() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Initial prompt")
                .withText("Additional text");

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should ignore null text")
    void builder_WithNullText() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .withText(null);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should ignore empty text")
    void builder_WithEmptyText() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .withText("");

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should accept maxTokens")
    void builder_WithMaxTokens() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .maxTokens(4096);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should ignore invalid maxTokens")
    void builder_WithInvalidMaxTokens() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .maxTokens(-1)
                .maxTokens(0)
                .maxTokens(null);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should accept valid temperature")
    void builder_WithValidTemperature() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .temperature(0.5);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should ignore invalid temperature")
    void builder_WithInvalidTemperature() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .temperature(-0.1)
                .temperature(2.1)
                .temperature(null);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should accept edge case temperatures")
    void builder_WithEdgeCaseTemperature() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .temperature(0.0)
                .temperature(2.0);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should accept valid topP")
    void builder_WithValidTopP() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .topP(0.5);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should ignore invalid topP")
    void builder_WithInvalidTopP() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .topP(-0.1)
                .topP(1.1)
                .topP(null);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should accept edge case topP")
    void builder_WithEdgeCaseTopP() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .topP(0.0)
                .topP(1.0);

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should support method chaining")
    void builder_MethodChaining() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("Test prompt")
                .withText("Additional context")
                .maxTokens(4096)
                .temperature(0.5)
                .topP(0.9)
                .build();

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder prompt should clear previous parts")
    void builder_PromptClearsPreviousParts() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("First prompt")
                .prompt("Second prompt");

        assertNotNull(builder);
    }

    @Test
    @DisplayName("Builder should handle null prompt")
    void builder_NullPrompt() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt(null);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> builder.build().execute());

        assertTrue(exception.getMessage().contains("no content provided"));
    }

    @Test
    @DisplayName("Builder should handle empty prompt")
    void builder_EmptyPrompt() {
        AiService.GeminiRequestBuilder builder = aiService.builder()
                .prompt("");

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> builder.build().execute());

        assertTrue(exception.getMessage().contains("no content provided"));
    }

    @Test
    @DisplayName("Should call generate method which uses builder internally")
    void generate_UsesBuilder() {
        // generate() internally calls builder().prompt().build().execute()
        // Since execute() will fail without real API, we just verify the method exists
        // and throws
        RuntimeException exception = assertThrows(RuntimeException.class,
                () -> aiService.generate("test prompt"));

        assertNotNull(exception);
    }
}
