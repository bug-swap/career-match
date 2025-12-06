package com.careermatch.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Response DTO for Gemini AI API
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class GeminiResponse {
    
    private List<Candidate> candidates;
    private UsageMetadata usageMetadata;
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Candidate {
        private Content content;
        private String finishReason;
        private Integer index;
        private List<SafetyRating> safetyRatings;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Content {
        private List<Part> parts;
        private String role;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class Part {
        private String text;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class SafetyRating {
        private String category;
        private String probability;
    }
    
    @Data
    @NoArgsConstructor
    @AllArgsConstructor
    public static class UsageMetadata {
        private Integer promptTokenCount;
        private Integer candidatesTokenCount;
        private Integer totalTokenCount;
    }
}