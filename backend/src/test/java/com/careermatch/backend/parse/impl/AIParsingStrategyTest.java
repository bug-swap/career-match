package com.careermatch.backend.parse.impl;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.AiService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class AIParsingStrategyTest {

    @Mock
    private AiService aiService;

    @Mock
    private AiService.GeminiRequestBuilder mockBuilder;

    private AIParsingStrategy aiParsingStrategy;
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        objectMapper = new ObjectMapper();
        aiParsingStrategy = new AIParsingStrategy(aiService, objectMapper);
    }

    @Test
    @DisplayName("Should return correct strategy name")
    void getStrategyName() {
        assertEquals("AI", aiParsingStrategy.getStrategyName());
    }

    @Test
    @DisplayName("Should parse resume successfully")
    void parse_Success() throws Exception {
        String aiResponse = """
                {
                    "success": true,
                    "sections": {
                        "education": [{"degree": "BS", "major": "CS", "institution": "MIT", "location": "Boston", "date": "2020", "gpa": "3.8"}],
                        "experience": [{"title": "Engineer", "company": "Google", "date": "2020-Present", "isPresent": true, "location": "CA", "responsibilities": ["Coding"]}],
                        "project": [],
                        "publication": [],
                        "contact": {"name": "John Doe", "email": "john@example.com", "phone": "123", "linkedin": "", "github": "", "website": ""}
                    },
                    "metadata": {"section_count": 3, "total_items": 3, "total_char_count": 1000, "processing_time_ms": 500}
                }
                """;

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "John Doe\nSoftware Engineer".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(aiResponse);

        Resume result = aiParsingStrategy.parse(txtFile);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertNotNull(result.getSections());
        assertEquals("John Doe", result.getSections().getContact().getName());
        assertEquals(1, result.getSections().getEducation().size());
        assertEquals(1, result.getSections().getExperience().size());
    }

    @Test
    @DisplayName("Should handle JSON response with markdown code blocks")
    void parse_WithMarkdownCodeBlocks() throws Exception {
        String aiResponse = """
                ```json
                {
                    "success": true,
                    "sections": {
                        "education": [],
                        "experience": [],
                        "project": [],
                        "publication": [],
                        "contact": {"name": "Jane Doe", "email": "", "phone": "", "linkedin": "", "github": "", "website": ""}
                    },
                    "metadata": {}
                }
                ```
                """;

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "Jane Doe".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(aiResponse);

        Resume result = aiParsingStrategy.parse(txtFile);

        assertNotNull(result);
        assertEquals("Jane Doe", result.getSections().getContact().getName());
    }

    @Test
    @DisplayName("Should handle JSON response with only opening code block marker")
    void parse_WithOnlyOpeningCodeBlock() throws Exception {
        String aiResponse = """
                ```
                {
                    "success": true,
                    "sections": {
                        "education": [],
                        "experience": [],
                        "project": [],
                        "publication": [],
                        "contact": {"name": "Test", "email": "", "phone": "", "linkedin": "", "github": "", "website": ""}
                    },
                    "metadata": {}
                }
                ```
                """;

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "Test".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(aiResponse);

        Resume result = aiParsingStrategy.parse(txtFile);

        assertNotNull(result);
    }

    @Test
    @DisplayName("Should throw exception on invalid JSON structure")
    void parse_ThrowsExceptionOnInvalidJsonStructure() throws Exception {
        String invalidJson = "This is not valid JSON at all";

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "Content".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(invalidJson);

        Exception exception = assertThrows(Exception.class, () -> {
            aiParsingStrategy.parse(txtFile);
        });

        assertTrue(exception.getMessage().contains("Invalid JSON structure"));
    }

    @Test
    @DisplayName("Should create fallback resume on JSON parsing error")
    void parse_FallbackOnJsonParsingError() throws Exception {
        // Valid JSON structure but wrong schema that fails to deserialize properly
        String malformedJson = """
                {
                    "success": "not_a_boolean",
                    "sections": "not_an_object",
                    "metadata": []
                }
                """;

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "Content".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(malformedJson);

        Resume result = aiParsingStrategy.parse(txtFile);

        assertNotNull(result);
        assertFalse(result.isSuccess());
        assertNotNull(result.getSections());
        assertNotNull(result.getMetadata());
        assertEquals(0, result.getMetadata().getSectionCount());
    }

    @Test
    @DisplayName("Should enrich metadata correctly")
    void parse_EnrichMetadata() throws Exception {
        String aiResponse = """
                {
                    "success": true,
                    "sections": {
                        "education": [{"degree": "BS", "major": "CS", "institution": "MIT", "location": "", "date": "", "gpa": ""}],
                        "experience": [{"title": "Dev", "company": "X", "date": "", "isPresent": false, "location": "", "responsibilities": []}],
                        "project": [{"name": "Project", "url": "", "description": [], "technologies": "", "date": ""}],
                        "publication": [{"title": "Paper", "year": "2023", "doi": ""}],
                        "contact": {"name": "Test", "email": "", "phone": "", "linkedin": "", "github": "", "website": ""}
                    }
                }
                """;

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "Test content here".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(aiResponse);

        Resume result = aiParsingStrategy.parse(txtFile);

        assertNotNull(result);
        assertNotNull(result.getMetadata());
        assertEquals(5, result.getMetadata().getSectionCount());
        assertEquals(4, result.getMetadata().getTotalItems());
        assertTrue(result.getMetadata().getTotalCharCount() > 0);
        assertTrue(result.getMetadata().getProcessingTimeMs() >= 0);
    }

    @Test
    @DisplayName("Should handle response with null metadata")
    void parse_NullMetadata() throws Exception {
        String aiResponse = """
                {
                    "success": true,
                    "sections": {
                        "education": [],
                        "experience": [],
                        "project": [],
                        "publication": [],
                        "contact": {"name": "Test", "email": "", "phone": "", "linkedin": "", "github": "", "website": ""}
                    }
                }
                """;

        MockMultipartFile txtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "Test".getBytes());

        when(aiService.builder()).thenReturn(mockBuilder);
        when(mockBuilder.prompt(anyString())).thenReturn(mockBuilder);
        when(mockBuilder.maxTokens(anyInt())).thenReturn(mockBuilder);
        when(mockBuilder.temperature(anyDouble())).thenReturn(mockBuilder);
        when(mockBuilder.build()).thenReturn(mockBuilder);
        when(mockBuilder.execute()).thenReturn(aiResponse);

        Resume result = aiParsingStrategy.parse(txtFile);

        assertNotNull(result);
        assertNotNull(result.getMetadata());
    }

    @Test
    @DisplayName("Should throw exception for unsupported file format")
    void parse_UnsupportedFileFormat() {
        MockMultipartFile unsupportedFile = new MockMultipartFile(
                "file",
                "resume.xyz",
                "application/octet-stream",
                "Content".getBytes());

        assertThrows(Exception.class, () -> aiParsingStrategy.parse(unsupportedFile));
    }
}
