package com.careermatch.backend.service;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.parse.ResumeParserFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class ResumeParserServiceTest {

    @Mock
    private ResumeParserFactory resumeParserFactory;

    @InjectMocks
    private ResumeParserService resumeParserService;

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
                                .build())
                        .education(Collections.emptyList())
                        .experience(Collections.emptyList())
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .build())
                .metadata(Resume.Metadata.builder()
                        .sectionCount(1)
                        .totalItems(1)
                        .build())
                .build();
    }

    @Test
    @DisplayName("Should parse resume with AI strategy")
    void parseResume_AIStrategy_Success() {
        when(resumeParserFactory.parse(any(MultipartFile.class), eq(ParsingStrategyType.AI)))
                .thenReturn(mockResume);

        Resume result = resumeParserService.parseResume(mockResumeFile, ParsingStrategyType.AI);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertEquals("John Doe", result.getSections().getContact().getName());

        verify(resumeParserFactory).parse(mockResumeFile, ParsingStrategyType.AI);
    }

    @Test
    @DisplayName("Should parse resume with ML strategy")
    void parseResume_MLStrategy_Success() {
        when(resumeParserFactory.parse(any(MultipartFile.class), eq(ParsingStrategyType.ML)))
                .thenReturn(mockResume);

        Resume result = resumeParserService.parseResume(mockResumeFile, ParsingStrategyType.ML);

        assertNotNull(result);
        assertTrue(result.isSuccess());

        verify(resumeParserFactory).parse(mockResumeFile, ParsingStrategyType.ML);
    }

    @Test
    @DisplayName("Should parse resume with null strategy (default)")
    void parseResume_NullStrategy_Success() {
        when(resumeParserFactory.parse(any(MultipartFile.class), isNull()))
                .thenReturn(mockResume);

        Resume result = resumeParserService.parseResume(mockResumeFile, null);

        assertNotNull(result);
        assertTrue(result.isSuccess());

        verify(resumeParserFactory).parse(mockResumeFile, null);
    }

    @Test
    @DisplayName("Should propagate exception from factory")
    void parseResume_FactoryThrowsException() {
        when(resumeParserFactory.parse(any(MultipartFile.class), any()))
                .thenThrow(new RuntimeException("Parsing failed"));

        RuntimeException exception = assertThrows(RuntimeException.class,
                () -> resumeParserService.parseResume(mockResumeFile, ParsingStrategyType.AI));

        assertEquals("Parsing failed", exception.getMessage());
    }
}
