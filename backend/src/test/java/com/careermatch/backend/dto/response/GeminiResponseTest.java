package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GeminiResponseTest {

    @Test
    void testGeminiResponse() {
        GeminiResponse response = new GeminiResponse();
        assertNotNull(response);

        List<GeminiResponse.Candidate> candidates = Arrays.asList(new GeminiResponse.Candidate());
        GeminiResponse.UsageMetadata metadata = new GeminiResponse.UsageMetadata();

        response.setCandidates(candidates);
        response.setUsageMetadata(metadata);

        assertEquals(candidates, response.getCandidates());
        assertEquals(metadata, response.getUsageMetadata());
    }

    @Test
    void testGeminiResponseAllArgs() {
        List<GeminiResponse.Candidate> candidates = Arrays.asList(new GeminiResponse.Candidate());
        GeminiResponse.UsageMetadata metadata = new GeminiResponse.UsageMetadata(10, 20, 30);

        GeminiResponse response = new GeminiResponse(candidates, metadata);

        assertEquals(candidates, response.getCandidates());
        assertEquals(metadata, response.getUsageMetadata());
    }

    @Test
    void testCandidate() {
        GeminiResponse.Candidate candidate = new GeminiResponse.Candidate();
        GeminiResponse.Content content = new GeminiResponse.Content();
        List<GeminiResponse.SafetyRating> ratings = Arrays.asList(new GeminiResponse.SafetyRating());

        candidate.setContent(content);
        candidate.setFinishReason("STOP");
        candidate.setIndex(0);
        candidate.setSafetyRatings(ratings);

        assertEquals(content, candidate.getContent());
        assertEquals("STOP", candidate.getFinishReason());
        assertEquals(0, candidate.getIndex());
        assertEquals(ratings, candidate.getSafetyRatings());

        GeminiResponse.Candidate candidate2 = new GeminiResponse.Candidate(content, "STOP", 1, ratings);
        assertEquals(content, candidate2.getContent());
    }

    @Test
    void testContent() {
        GeminiResponse.Content content = new GeminiResponse.Content();
        List<GeminiResponse.Part> parts = Arrays.asList(new GeminiResponse.Part("text"));

        content.setParts(parts);
        content.setRole("user");

        assertEquals(parts, content.getParts());
        assertEquals("user", content.getRole());

        GeminiResponse.Content content2 = new GeminiResponse.Content(parts, "model");
        assertEquals("model", content2.getRole());
    }

    @Test
    void testPart() {
        GeminiResponse.Part part = new GeminiResponse.Part();
        part.setText("hello");
        assertEquals("hello", part.getText());

        GeminiResponse.Part part2 = new GeminiResponse.Part("world");
        assertEquals("world", part2.getText());
    }

    @Test
    void testSafetyRating() {
        GeminiResponse.SafetyRating rating = new GeminiResponse.SafetyRating();
        rating.setCategory("HARM_CATEGORY_DANGEROUS");
        rating.setProbability("LOW");

        assertEquals("HARM_CATEGORY_DANGEROUS", rating.getCategory());
        assertEquals("LOW", rating.getProbability());

        GeminiResponse.SafetyRating rating2 = new GeminiResponse.SafetyRating("CAT", "HIGH");
        assertEquals("CAT", rating2.getCategory());
        assertEquals("HIGH", rating2.getProbability());
    }

    @Test
    void testUsageMetadata() {
        GeminiResponse.UsageMetadata metadata = new GeminiResponse.UsageMetadata();
        metadata.setPromptTokenCount(100);
        metadata.setCandidatesTokenCount(200);
        metadata.setTotalTokenCount(300);

        assertEquals(100, metadata.getPromptTokenCount());
        assertEquals(200, metadata.getCandidatesTokenCount());
        assertEquals(300, metadata.getTotalTokenCount());
    }
}
