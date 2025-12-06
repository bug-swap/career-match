package com.careermatch.backend.dto.request;

import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class GeminiRequestTest {

    @Test
    void testGeminiRequest() {
        GeminiRequest request = new GeminiRequest();
        assertNotNull(request);

        List<GeminiRequest.Content> contents = Arrays.asList(new GeminiRequest.Content());
        GeminiRequest.GenerationConfig config = new GeminiRequest.GenerationConfig();

        request.setContents(contents);
        request.setGenerationConfig(config);

        assertEquals(contents, request.getContents());
        assertEquals(config, request.getGenerationConfig());
    }

    @Test
    void testGeminiRequestAllArgs() {
        GeminiRequest.GenerationConfig config = new GeminiRequest.GenerationConfig(100, 0.7, 0.9, 0.0, 0.0);
        List<GeminiRequest.Content> contents = Arrays.asList(new GeminiRequest.Content());

        GeminiRequest request = new GeminiRequest(contents, config);

        assertEquals(contents, request.getContents());
        assertEquals(config, request.getGenerationConfig());
    }

    @Test
    void testContent() {
        GeminiRequest.Content content = new GeminiRequest.Content();
        List<GeminiRequest.Part> parts = Arrays.asList(new GeminiRequest.Part("test"));

        content.setParts(parts);
        assertEquals(parts, content.getParts());

        GeminiRequest.Content content2 = new GeminiRequest.Content(parts);
        assertEquals(parts, content2.getParts());
    }

    @Test
    void testPart() {
        GeminiRequest.Part part = new GeminiRequest.Part();
        part.setText("hello");
        assertEquals("hello", part.getText());

        GeminiRequest.InlineData inlineData = new GeminiRequest.InlineData("image/png", "base64data");
        part.setInlineData(inlineData);
        assertEquals(inlineData, part.getInlineData());

        GeminiRequest.Part part2 = new GeminiRequest.Part("text");
        assertEquals("text", part2.getText());

        GeminiRequest.Part part3 = new GeminiRequest.Part("t", inlineData);
        assertEquals("t", part3.getText());
        assertEquals(inlineData, part3.getInlineData());
    }

    @Test
    void testInlineData() {
        GeminiRequest.InlineData data = new GeminiRequest.InlineData();
        data.setMimeType("image/jpeg");
        data.setData("abc123");

        assertEquals("image/jpeg", data.getMimeType());
        assertEquals("abc123", data.getData());
    }

    @Test
    void testGenerationConfig() {
        GeminiRequest.GenerationConfig config = new GeminiRequest.GenerationConfig();
        config.setMaxOutputTokens(1000);
        config.setTemperature(0.5);
        config.setTopP(0.8);
        config.setFrequencyPenalty(0.1);
        config.setPresencePenalty(0.2);

        assertEquals(1000, config.getMaxOutputTokens());
        assertEquals(0.5, config.getTemperature());
        assertEquals(0.8, config.getTopP());
        assertEquals(0.1, config.getFrequencyPenalty());
        assertEquals(0.2, config.getPresencePenalty());
    }
}
