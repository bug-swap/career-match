package com.careermatch.backend.controller;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.ResumeParserService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Collections;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(ResumeController.class)
class ResumeControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private ResumeParserService resumeParserService;

    private Resume createMockResume() {
        return Resume.builder()
                .success(true)
                .sections(Resume.ResumeSection.builder()
                        .contact(Resume.Contact.builder()
                                .name("John Doe")
                                .email("john@example.com")
                                .build())
                        .education(Collections.singletonList(
                                Resume.Education.builder()
                                        .degree("BS")
                                        .major("CS")
                                        .institution("MIT")
                                        .build()))
                        .experience(Collections.emptyList())
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .build())
                .metadata(Resume.Metadata.builder()
                        .sectionCount(2)
                        .totalItems(2)
                        .totalCharCount(1000)
                        .processingTimeMs(500L)
                        .build())
                .build();
    }

    @Test
    @DisplayName("POST /api/v1/resume/parse - Should parse resume with default strategy")
    void parseResume_DefaultStrategy_Success() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(resumeParserService.parseResume(any(), isNull()))
                .thenReturn(createMockResume());

        mockMvc.perform(multipart("/api/v1/resume/parse")
                .file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.sections.contact.name").value("John Doe"))
                .andExpect(jsonPath("$.sections.contact.email").value("john@example.com"))
                .andExpect(jsonPath("$.sections.education[0].degree").value("BS"));
    }

    @Test
    @DisplayName("POST /api/v1/resume/parse/AI - Should parse resume with AI strategy")
    void parseResume_AIStrategy_Success() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(resumeParserService.parseResume(any(), eq(ParsingStrategyType.AI)))
                .thenReturn(createMockResume());

        mockMvc.perform(multipart("/api/v1/resume/parse/AI")
                .file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));
    }

    @Test
    @DisplayName("POST /api/v1/resume/parse/ML - Should parse resume with ML strategy")
    void parseResume_MLStrategy_Success() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(resumeParserService.parseResume(any(), eq(ParsingStrategyType.ML)))
                .thenReturn(createMockResume());

        mockMvc.perform(multipart("/api/v1/resume/parse/ML")
                .file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true));
    }

    @Test
    @DisplayName("POST /api/v1/resume/parse - Should return 400 on validation error")
    void parseResume_ValidationError() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(resumeParserService.parseResume(any(), any()))
                .thenThrow(new IllegalArgumentException("Invalid file format"));

        mockMvc.perform(multipart("/api/v1/resume/parse")
                .file(file))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.error").value("Validation Error"))
                .andExpect(jsonPath("$.message").value("Invalid file format"));
    }

    @Test
    @DisplayName("POST /api/v1/resume/parse - Should return 500 on parsing error")
    void parseResume_ParsingError() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(resumeParserService.parseResume(any(), any()))
                .thenThrow(new RuntimeException("Parsing failed"));

        mockMvc.perform(multipart("/api/v1/resume/parse")
                .file(file))
                .andExpect(status().isInternalServerError())
                .andExpect(jsonPath("$.error").value("Parsing Error"));
    }

    @Test
    @DisplayName("GET /api/v1/resume/health - Should return health status")
    void health() throws Exception {
        mockMvc.perform(get("/api/v1/resume/health"))
                .andExpect(status().isOk())
                .andExpect(content().string("Resume Parser Service is running"));
    }
}
