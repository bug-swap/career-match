package com.careermatch.backend.parse.impl;

import com.careermatch.backend.dto.response.SectionsResponse;
import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.PythonMLService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class MLParsingStrategyTest {

    @Mock
    private PythonMLService pythonMLService;

    @InjectMocks
    private MLParsingStrategy mlParsingStrategy;

    private MultipartFile mockTxtFile;

    @BeforeEach
    void setUp() {
        mockTxtFile = new MockMultipartFile(
                "file",
                "resume.txt",
                "text/plain",
                "John Doe\nSoftware Engineer at Google".getBytes());
    }

    @Test
    @DisplayName("Should return correct strategy name")
    void getStrategyName() {
        assertEquals("ML", mlParsingStrategy.getStrategyName());
    }

    @Test
    @DisplayName("Should parse resume successfully with all sections")
    void parse_Success_AllSections() throws Exception {
        SectionsResponse sectionsResponse = createFullSectionsResponse();
        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertNotNull(result.getSections());

        // Verify contact
        assertEquals("John Doe", result.getSections().getContact().getName());
        assertEquals("john@example.com", result.getSections().getContact().getEmail());

        // Verify education
        assertEquals(1, result.getSections().getEducation().size());
        assertEquals("Bachelor of Science", result.getSections().getEducation().get(0).getDegree());

        // Verify experience
        assertEquals(1, result.getSections().getExperience().size());
        assertEquals("Software Engineer", result.getSections().getExperience().get(0).getTitle());
        assertTrue(result.getSections().getExperience().get(0).getIsPresent());

        // Verify projects
        assertEquals(1, result.getSections().getProject().size());
        assertEquals("Resume Parser", result.getSections().getProject().get(0).getName());

        // Verify publications
        assertEquals(1, result.getSections().getPublication().size());
        assertEquals("ML Paper", result.getSections().getPublication().get(0).getTitle());

        verify(pythonMLService).extractSections(mockTxtFile);
    }

    @Test
    @DisplayName("Should handle response with null sections")
    void parse_NullSections() throws Exception {
        SectionsResponse sectionsResponse = SectionsResponse.builder()
                .success(true)
                .sections(null)
                .build();

        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertTrue(result.isSuccess());
        assertNotNull(result.getSections());
        assertNotNull(result.getSections().getEducation());
        assertTrue(result.getSections().getEducation().isEmpty());
    }

    @Test
    @DisplayName("Should handle response with empty education list")
    void parse_EmptyEducation() throws Exception {
        SectionsResponse sectionsResponse = SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder()
                        .education(null)
                        .experience(Collections.emptyList())
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .contact(null)
                        .build())
                .build();

        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertNotNull(result.getSections().getEducation());
        assertTrue(result.getSections().getEducation().isEmpty());
    }

    @Test
    @DisplayName("Should handle response with null contact")
    void parse_NullContact() throws Exception {
        SectionsResponse sectionsResponse = SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder()
                        .education(Collections.emptyList())
                        .experience(Collections.emptyList())
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .contact(null)
                        .build())
                .build();

        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertNotNull(result.getSections().getContact());
    }

    @Test
    @DisplayName("Should handle experience with null responsibilities")
    void parse_ExperienceWithNullResponsibilities() throws Exception {
        SectionsResponse.Experience exp = SectionsResponse.Experience.builder()
                .title("Engineer")
                .company("Company")
                .responsibilities(null)
                .build();

        SectionsResponse sectionsResponse = SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder()
                        .education(Collections.emptyList())
                        .experience(Collections.singletonList(exp))
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .contact(null)
                        .build())
                .build();

        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertNotNull(result.getSections().getExperience().get(0).getResponsibilities());
        assertTrue(result.getSections().getExperience().get(0).getResponsibilities().isEmpty());
    }

    @Test
    @DisplayName("Should handle project with null description")
    void parse_ProjectWithNullDescription() throws Exception {
        SectionsResponse.Project proj = SectionsResponse.Project.builder()
                .name("Project")
                .description(null)
                .build();

        SectionsResponse sectionsResponse = SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder()
                        .education(Collections.emptyList())
                        .experience(Collections.emptyList())
                        .project(Collections.singletonList(proj))
                        .publication(Collections.emptyList())
                        .contact(null)
                        .build())
                .build();

        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertNotNull(result.getSections().getProject().get(0).getDescription());
        assertTrue(result.getSections().getProject().get(0).getDescription().isEmpty());
    }

    @Test
    @DisplayName("Should calculate metadata correctly")
    void parse_MetadataCalculation() throws Exception {
        SectionsResponse sectionsResponse = createFullSectionsResponse();
        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertNotNull(result.getMetadata());
        assertEquals(5, result.getMetadata().getSectionCount());
        assertEquals(4, result.getMetadata().getTotalItems());
        assertTrue(result.getMetadata().getTotalCharCount() > 0);
        assertTrue(result.getMetadata().getProcessingTimeMs() >= 0);
    }

    @Test
    @DisplayName("Should handle metadata with null resume metadata")
    void parse_NullResumeMetadata() throws Exception {
        SectionsResponse sectionsResponse = SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder()
                        .education(Collections.emptyList())
                        .experience(Collections.emptyList())
                        .project(Collections.emptyList())
                        .publication(Collections.emptyList())
                        .contact(null)
                        .build())
                .metadata(null)
                .build();

        when(pythonMLService.extractSections(any(MultipartFile.class))).thenReturn(sectionsResponse);

        Resume result = mlParsingStrategy.parse(mockTxtFile);

        assertNotNull(result);
        assertNotNull(result.getMetadata());
    }

    @Test
    @DisplayName("Should propagate exception from service")
    void parse_ServiceThrowsException() {
        when(pythonMLService.extractSections(any(MultipartFile.class)))
                .thenThrow(new RuntimeException("ML Service error"));

        assertThrows(RuntimeException.class, () -> mlParsingStrategy.parse(mockTxtFile));
    }

    @Test
    @DisplayName("Should handle unsupported file format")
    void parse_UnsupportedFileFormat() {
        MockMultipartFile unsupportedFile = new MockMultipartFile(
                "file",
                "resume.xyz",
                "application/octet-stream",
                "Content".getBytes());

        assertThrows(Exception.class, () -> mlParsingStrategy.parse(unsupportedFile));
    }

    private SectionsResponse createFullSectionsResponse() {
        return SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder()
                        .education(Collections.singletonList(
                                SectionsResponse.Education.builder()
                                        .degree("Bachelor of Science")
                                        .major("Computer Science")
                                        .institution("MIT")
                                        .location("Boston, MA")
                                        .date("2020")
                                        .gpa("3.9")
                                        .build()))
                        .experience(Collections.singletonList(
                                SectionsResponse.Experience.builder()
                                        .title("Software Engineer")
                                        .company("Google")
                                        .date("01/2020 - Present")
                                        .isPresent(true)
                                        .location("Mountain View, CA")
                                        .responsibilities(Arrays.asList("Develop features", "Code reviews"))
                                        .build()))
                        .project(Collections.singletonList(
                                SectionsResponse.Project.builder()
                                        .name("Resume Parser")
                                        .url("https://github.com/example/resume-parser")
                                        .description(Collections.singletonList("AI-powered resume parsing"))
                                        .technologies("Java, Spring Boot, ML")
                                        .date("2023")
                                        .build()))
                        .publication(Collections.singletonList(
                                SectionsResponse.Publication.builder()
                                        .title("ML Paper")
                                        .year("2023")
                                        .doi("10.1234/example")
                                        .build()))
                        .contact(SectionsResponse.Contact.builder()
                                .name("John Doe")
                                .email("john@example.com")
                                .phone("123-456-7890")
                                .linkedin("linkedin.com/in/johndoe")
                                .github("github.com/johndoe")
                                .website("johndoe.com")
                                .build())
                        .build())
                .build();
    }
}
