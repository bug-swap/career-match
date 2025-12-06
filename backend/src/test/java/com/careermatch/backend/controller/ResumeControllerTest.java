package com.careermatch.backend.controller;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.ResumeParserService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ResumeControllerTest {

    @Mock
    private ResumeParserService resumeParserService;

    @InjectMocks
    private ResumeController resumeController;

    private MultipartFile mockResumeFile;
    private Resume mockResume;

    @BeforeEach
    void setUp() {
        mockResumeFile = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test resume content".getBytes());

        mockResume = Resume.builder()
                .success(true)
                .sections(Resume.ResumeSection.builder()
                        .contact(Resume.Contact.builder()
                                .name("John Doe")
                                .email("john@example.com")
                                .phone("123-456-7890")
                                .build())
                        .education(Collections.singletonList(
                                Resume.Education.builder()
                                        .degree("Bachelor of Science")
                                        .major("Computer Science")
                                        .institution("MIT")
                                        .build()))
                        .experience(Collections.singletonList(
                                Resume.Experience.builder()
                                        .title("Software Engineer")
                                        .company("Google")
                                        .date("01/2020 - Present")
                                        .isPresent(true)
                                        .responsibilities(Arrays.asList("Developed features", "Code reviews"))
                                        .build()))
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .build())
                .metadata(Resume.Metadata.builder()
                        .sectionCount(3)
                        .totalItems(3)
                        .totalCharCount(1000)
                        .processingTimeMs(500L)
                        .build())
                .build();
    }

    @Test
    @DisplayName("Should parse resume with default strategy")
    void parseResume_DefaultStrategy_Success() {
        when(resumeParserService.parseResume(any(MultipartFile.class), isNull()))
                .thenReturn(mockResume);

        ResponseEntity<?> response = resumeController.parseResume(mockResumeFile);

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertTrue(response.getBody() instanceof Resume);

        Resume result = (Resume) response.getBody();
        assertTrue(result.isSuccess());
        assertEquals("John Doe", result.getSections().getContact().getName());

        verify(resumeParserService, times(1)).parseResume(any(MultipartFile.class), isNull());
    }

    @Test
    @DisplayName("Should parse resume with AI strategy")
    void parseResumeWithStrategy_AI_Success() {
        when(resumeParserService.parseResume(any(MultipartFile.class), eq(ParsingStrategyType.AI)))
                .thenReturn(mockResume);

        ResponseEntity<?> response = resumeController.parseResumeWithStrategy(mockResumeFile, "AI");

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());

        verify(resumeParserService, times(1)).parseResume(any(MultipartFile.class), eq(ParsingStrategyType.AI));
    }

    @Test
    @DisplayName("Should parse resume with ML strategy")
    void parseResumeWithStrategy_ML_Success() {
        when(resumeParserService.parseResume(any(MultipartFile.class), eq(ParsingStrategyType.ML)))
                .thenReturn(mockResume);

        ResponseEntity<?> response = resumeController.parseResumeWithStrategy(mockResumeFile, "ML");

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());

        verify(resumeParserService, times(1)).parseResume(any(MultipartFile.class), eq(ParsingStrategyType.ML));
    }

    @Test
    @DisplayName("Should return bad request on IllegalArgumentException")
    void parseResumeWithStrategy_ValidationError() {
        when(resumeParserService.parseResume(any(MultipartFile.class), any()))
                .thenThrow(new IllegalArgumentException("Invalid file format"));

        ResponseEntity<?> response = resumeController.parseResumeWithStrategy(mockResumeFile, "AI");

        assertNotNull(response);
        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
    }

    @Test
    @DisplayName("Should return internal server error on generic exception")
    void parseResumeWithStrategy_GenericException() {
        when(resumeParserService.parseResume(any(MultipartFile.class), any()))
                .thenThrow(new RuntimeException("Unexpected error"));

        ResponseEntity<?> response = resumeController.parseResumeWithStrategy(mockResumeFile, "AI");

        assertNotNull(response);
        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
    }

    @Test
    @DisplayName("Should handle null strategy and use default")
    void parseResumeWithStrategy_NullStrategy() {
        when(resumeParserService.parseResume(any(MultipartFile.class), isNull()))
                .thenReturn(mockResume);

        ResponseEntity<?> response = resumeController.parseResumeWithStrategy(mockResumeFile, null);

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());

        verify(resumeParserService).parseResume(any(MultipartFile.class), isNull());
    }

    @Test
    @DisplayName("Health check should return OK")
    void health_ReturnsOk() {
        ResponseEntity<String> response = resumeController.health();

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertEquals("Resume Parser Service is running", response.getBody());
    }

    @Test
    @DisplayName("Should handle lowercase strategy name")
    void parseResumeWithStrategy_LowercaseStrategy() {
        when(resumeParserService.parseResume(any(MultipartFile.class), eq(ParsingStrategyType.AI)))
                .thenReturn(mockResume);

        ResponseEntity<?> response = resumeController.parseResumeWithStrategy(mockResumeFile, "ai");

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());
    }
}
