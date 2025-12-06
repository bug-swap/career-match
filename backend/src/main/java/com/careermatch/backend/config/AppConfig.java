package com.careermatch.backend.config;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Application configuration
 */
@Configuration
public class AppConfig {

    /**
     * Configure ObjectMapper for JSON processing
     * @return configured ObjectMapper
     */
    @Bean
    public ObjectMapper objectMapper() {
        ObjectMapper mapper = new ObjectMapper();
        // Register JavaTimeModule for Java 8 date/time types
        mapper.registerModule(new JavaTimeModule());
        // Use snake_case for JSON property naming
        mapper.setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE);
        // Ignore unknown properties to handle schema mismatches
        mapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        return mapper;
    }
}