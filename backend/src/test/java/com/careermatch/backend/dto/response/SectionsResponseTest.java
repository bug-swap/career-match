package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class SectionsResponseTest {

    @Test
    @DisplayName("Should create SectionsResponse using builder")
    void builder() {
        SectionsResponse response = SectionsResponse.builder()
                .success(true)
                .sections(SectionsResponse.Sections.builder().build())
                .metadata(SectionsResponse.Metadata.builder().build())
                .build();

        assertTrue(response.isSuccess());
        assertNotNull(response.getSections());
        assertNotNull(response.getMetadata());
    }

    @Test
    @DisplayName("Should create SectionsResponse using no-args constructor")
    void noArgsConstructor() {
        SectionsResponse response = new SectionsResponse();

        assertFalse(response.isSuccess());
        assertNull(response.getSections());
        assertNull(response.getMetadata());
    }

    @Test
    @DisplayName("Should create Sections with all fields")
    void sections_AllFields() {
        SectionsResponse.Sections sections = SectionsResponse.Sections.builder()
                .education(Collections.singletonList(
                        SectionsResponse.Education.builder().degree("BS").build()))
                .experience(Collections.singletonList(
                        SectionsResponse.Experience.builder().title("Engineer").build()))
                .project(Collections.singletonList(
                        SectionsResponse.Project.builder().name("Project").build()))
                .publication(Collections.singletonList(
                        SectionsResponse.Publication.builder().title("Paper").build()))
                .contact(SectionsResponse.Contact.builder().name("John").build())
                .build();

        assertEquals(1, sections.getEducation().size());
        assertEquals(1, sections.getExperience().size());
        assertEquals(1, sections.getProject().size());
        assertEquals(1, sections.getPublication().size());
        assertEquals("John", sections.getContact().getName());
    }

    @Test
    @DisplayName("Should create Education using builder")
    void education_Builder() {
        SectionsResponse.Education education = SectionsResponse.Education.builder()
                .degree("Bachelor of Science")
                .major("Computer Science")
                .institution("MIT")
                .location("Boston, MA")
                .date("2020")
                .gpa("3.9")
                .build();

        assertEquals("Bachelor of Science", education.getDegree());
        assertEquals("Computer Science", education.getMajor());
        assertEquals("MIT", education.getInstitution());
        assertEquals("Boston, MA", education.getLocation());
        assertEquals("2020", education.getDate());
        assertEquals("3.9", education.getGpa());
    }

    @Test
    @DisplayName("Should create Experience using builder")
    void experience_Builder() {
        SectionsResponse.Experience experience = SectionsResponse.Experience.builder()
                .title("Software Engineer")
                .company("Google")
                .date("2020 - Present")
                .isPresent(true)
                .location("Mountain View")
                .responsibilities(Arrays.asList("Code", "Review"))
                .build();

        assertEquals("Software Engineer", experience.getTitle());
        assertEquals("Google", experience.getCompany());
        assertEquals("2020 - Present", experience.getDate());
        assertTrue(experience.getIsPresent());
        assertEquals("Mountain View", experience.getLocation());
        assertEquals(2, experience.getResponsibilities().size());
    }

    @Test
    @DisplayName("Should create Project using builder")
    void project_Builder() {
        SectionsResponse.Project project = SectionsResponse.Project.builder()
                .name("Resume Parser")
                .url("https://github.com/example")
                .description(Arrays.asList("Feature 1", "Feature 2"))
                .technologies("Java, Spring")
                .date("2023")
                .build();

        assertEquals("Resume Parser", project.getName());
        assertEquals("https://github.com/example", project.getUrl());
        assertEquals(2, project.getDescription().size());
        assertEquals("Java, Spring", project.getTechnologies());
        assertEquals("2023", project.getDate());
    }

    @Test
    @DisplayName("Should create Publication using builder")
    void publication_Builder() {
        SectionsResponse.Publication publication = SectionsResponse.Publication.builder()
                .title("ML Paper")
                .year("2023")
                .doi("10.1234/example")
                .build();

        assertEquals("ML Paper", publication.getTitle());
        assertEquals("2023", publication.getYear());
        assertEquals("10.1234/example", publication.getDoi());
    }

    @Test
    @DisplayName("Should create Contact using builder")
    void contact_Builder() {
        SectionsResponse.Contact contact = SectionsResponse.Contact.builder()
                .name("John Doe")
                .email("john@example.com")
                .phone("123-456-7890")
                .linkedin("linkedin.com/in/johndoe")
                .github("github.com/johndoe")
                .website("johndoe.com")
                .build();

        assertEquals("John Doe", contact.getName());
        assertEquals("john@example.com", contact.getEmail());
        assertEquals("123-456-7890", contact.getPhone());
        assertEquals("linkedin.com/in/johndoe", contact.getLinkedin());
        assertEquals("github.com/johndoe", contact.getGithub());
        assertEquals("johndoe.com", contact.getWebsite());
    }

    @Test
    @DisplayName("Should create Metadata using builder")
    void metadata_Builder() {
        SectionsResponse.Metadata metadata = SectionsResponse.Metadata.builder()
                .sectionCount(5)
                .totalItems(10)
                .totalCharCount(5000)
                .processingTimeMs(250L)
                .build();

        assertEquals(5, metadata.getSectionCount());
        assertEquals(10, metadata.getTotalItems());
        assertEquals(5000, metadata.getTotalCharCount());
        assertEquals(250L, metadata.getProcessingTimeMs());
    }

    @Test
    @DisplayName("Should handle null values in nested classes")
    void nullValues() {
        SectionsResponse.Education education = new SectionsResponse.Education();
        assertNull(education.getDegree());

        SectionsResponse.Experience experience = new SectionsResponse.Experience();
        assertNull(experience.getTitle());

        SectionsResponse.Contact contact = new SectionsResponse.Contact();
        assertNull(contact.getName());
    }

    @Test
    @DisplayName("Should use setters in nested classes")
    void setters() {
        SectionsResponse.Education education = new SectionsResponse.Education();
        education.setDegree("MS");
        education.setMajor("AI");
        assertEquals("MS", education.getDegree());
        assertEquals("AI", education.getMajor());

        SectionsResponse.Contact contact = new SectionsResponse.Contact();
        contact.setName("Jane");
        contact.setEmail("jane@test.com");
        assertEquals("Jane", contact.getName());
        assertEquals("jane@test.com", contact.getEmail());
    }
}
