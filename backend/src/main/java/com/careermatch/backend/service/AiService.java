package com.careermatch.backend.service;

import com.google.genai.Client;
import com.google.genai.types.Content;
import com.google.genai.types.GenerateContentConfig;
import com.google.genai.types.GenerateContentResponse;
import com.google.genai.types.Part;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * Service for interacting with Gemini AI API using google-genai SDK with builder pattern
 */
@Slf4j
@Service
public class AiService {

    @Value("${gemini.apikey}")
    private String apiKey;

    @Value("${gemini.model.name}")
    private String modelName;

    @Value("${gemini.max.tokens}")
    private Integer maxTokens;

    @Value("${gemini.temperature}")
    private Double temperature;

    @Value("${gemini.top.p}")
    private Double topP;

    private Client client;

    private void initializeClient() {
        if (client == null) {
            if (apiKey == null || apiKey.isEmpty() || apiKey.equals("your-api-key-here")) {
                log.error("❌ Gemini API key is not configured!");
                log.error("   Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable");
                throw new IllegalArgumentException("Gemini API key is missing or invalid");
            }

            client = Client.builder().apiKey(apiKey).build();
            log.info("✓ Initialized Google GenAI client with model: {}", modelName);
        }
    }

    /**
     * Generate content using Gemini AI with default configuration
     * @param prompt the input prompt
     * @return generated text response
     */
    public String generate(String prompt) {
        return builder()
                .prompt(prompt)
                .build()
                .execute();
    }

    /**
     * Create a new AI service request builder for fluent API
     * @return GeminiRequestBuilder instance
     */
    public GeminiRequestBuilder builder() {
        initializeClient();
        return new GeminiRequestBuilder(this);
    }

    /**
     * Builder class for fluent API to construct and execute Gemini AI requests
     */
    public static class GeminiRequestBuilder {
        private final AiService aiService;
        private final List<Part> parts;
        private Integer maxOutputTokens;
        private Double temperature;
        private Double topP;

        public GeminiRequestBuilder(AiService aiService) {
            this.aiService = aiService;
            this.parts = new ArrayList<>();
            // Set defaults from AiService config
            this.maxOutputTokens = aiService.maxTokens;
            this.temperature = aiService.temperature;
            this.topP = aiService.topP;
        }

        /**
         * Set text prompt - clears previous parts
         * @param prompt the text prompt
         * @return this builder for chaining
         */
        public GeminiRequestBuilder prompt(String prompt) {
            this.parts.clear();
            if (prompt != null && !prompt.isEmpty()) {
                this.parts.add(Part.fromText(prompt));
                log.debug("Prompt set, length: {} characters", prompt.length());
            }
            return this;
        }

        /**
         * Add additional text to the request (useful after adding files for captions/instructions)
         * @param text the text to add
         * @return this builder for chaining
         */
        public GeminiRequestBuilder withText(String text) {
            if (text != null && !text.isEmpty()) {
                this.parts.add(Part.fromText(text));
                log.debug("Text added, length: {} characters", text.length());
            }
            return this;
        }

        /**
         * Set maximum output tokens for the response
         * @param tokens max output tokens (higher = longer responses)
         * @return this builder for chaining
         */
        public GeminiRequestBuilder maxTokens(Integer tokens) {
            if (tokens != null && tokens > 0) {
                this.maxOutputTokens = tokens;
            }
            return this;
        }

        /**
         * Set temperature for response randomness
         * @param temp temperature value (0.0-2.0, lower = more deterministic)
         * @return this builder for chaining
         */
        public GeminiRequestBuilder temperature(Double temp) {
            if (temp != null && temp >= 0.0 && temp <= 2.0) {
                this.temperature = temp;
            }
            return this;
        }

        /**
         * Set top-p (nucleus sampling) parameter
         * @param p top-p value (0.0-1.0)
         * @return this builder for chaining
         */
        public GeminiRequestBuilder topP(Double p) {
            if (p != null && p >= 0.0 && p <= 1.0) {
                this.topP = p;
            }
            return this;
        }

        /**
         * Finalize the builder configuration
         * @return this builder (for method chaining consistency)
         */
        public GeminiRequestBuilder build() {
            log.debug("Request built with {} parts, maxTokens={}, temperature={}, topP={}",
                    parts.size(), maxOutputTokens, temperature, topP);
            return this;
        }

        /**
         * Execute the request to the Gemini API and return the generated text
         * @return the generated text response from Gemini
         * @throws RuntimeException if the API call fails
         */
        public String execute() {
            if (parts.isEmpty()) {
                throw new IllegalStateException("Cannot execute: no content provided. Use .prompt() or .addFile() first.");
            }

            try {
                log.info("Executing Gemini request with {} parts", parts.size());

                // Build content from parts
                Content content = Content.fromParts(parts.toArray(new Part[0]));

                // Build generation config with fluent builder
                GenerateContentConfig config = GenerateContentConfig.builder()
                        .maxOutputTokens(maxOutputTokens)
                        .temperature(temperature.floatValue())
                        .topP(topP.floatValue())
                        .build();

                log.debug("Sending to model: {} with config: maxTokens={}, temp={}, topP={}",
                        aiService.modelName, maxOutputTokens, temperature, topP);

                // Call Gemini API
                GenerateContentResponse response = aiService.client.models.generateContent(
                        aiService.modelName,
                        content,
                        config
                );
                log.info("Received response from Gemini API {}", response);

                String result = response.text();
                log.info("Request completed successfully. Response length: {} characters", result.length());

                return result;

            } catch (Exception e) {
                log.error("Failed to execute Gemini request: {}", e.getMessage(), e);
                log.info("ApiKey: {}, Model: {}", aiService.apiKey, aiService.modelName);
                throw new RuntimeException("Failed to generate AI response: " + e.getMessage(), e);
            }
        }
    }
}
