package com.careermatch.backend.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

import jakarta.annotation.PostConstruct;

/**
 * Configuration for Gemini AI API keys and settings
 */
@Slf4j
@Configuration
public class GeminiConfig {

    @Value("${gemini.apikey}")
    private String apiKey;

    @Value("${gemini.model.name}")
    private String modelName;

    @PostConstruct
    public void validateConfiguration() {
        log.info("========== Gemini Configuration Validation ==========");

        // Check API Key
        if (apiKey == null || apiKey.isEmpty() || apiKey.equals("your-api-key-here")) {
            log.error("❌ GEMINI API KEY NOT CONFIGURED!");
            log.error("   Set one of these environment variables:");
            log.error("   - GOOGLE_API_KEY");
            log.error("   - GEMINI_API_KEY");
            log.error("   OR update application.properties with your key");
            throw new IllegalArgumentException("Gemini API key is not configured. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.");
        }

        log.info("✓ API Key loaded (length: {})", apiKey.length());
        log.info("✓ Model name: {}", modelName);
        log.info("=====================================================");
    }
}

