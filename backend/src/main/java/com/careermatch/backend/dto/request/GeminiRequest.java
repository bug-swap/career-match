package com.careermatch.backend.dto.request;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Request DTO for Gemini AI API
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class GeminiRequest {
    
    private List<Content> contents;
    
    @JsonProperty("generationConfig")
    private GenerationConfig generationConfig;
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Content {
        private List<Part> parts;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static class Part {
        private String text;

        @JsonProperty("inlineData")
        private InlineData inlineData;

        public Part(String text) {
            this.text = text;
        }
    }

    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class InlineData {
        @JsonProperty("mimeType")
        private String mimeType;

        private String data;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class GenerationConfig {
        @JsonProperty("maxOutputTokens")
        private Integer maxOutputTokens;
        
        private Double temperature;
        
        @JsonProperty("topP")
        private Double topP;
        
        @JsonProperty("frequencyPenalty")
        private Double frequencyPenalty;
        
        @JsonProperty("presencePenalty")
        private Double presencePenalty;
    }
}