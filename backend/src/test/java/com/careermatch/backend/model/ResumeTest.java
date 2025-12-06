package com.careermatch.backend.model;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.*;

class ResumeTest {

    @Test
    @DisplayName("Should create Resume using builder")
    void builder() {
        Resume resume = Resume.builder()
                .success(true)
                .sections(Resume.ResumeSection.builder().build())
                .metadata(Resume.Metadata.builder().build())
                .build();

        assertTrue(resume.isSuccess());
        assertNotNull(resume.getSections());
        assertNotNull(resume.getMetadata());
    }

    @Test
    @DisplayName("Should create Resume using no-args constructor")
    void noArgsConstructor() {
        Resume resume = new Resume();

        assertFalse(resume.isSuccess());
        assertNull(resume.getSections());
        assertNull(resume.getMetadata());
    }

    @Test
    @DisplayName("Should create Resume using all-args constructor")
    void allArgsConstructor() {
        Resume.ResumeSection sections = Resume.ResumeSection.builder().build();
        Resume.Metadata metadata = Resume.Metadata.builder().build();

        Resume resume = new Resume(true, sections, metadata);

        assertTrue(resume.isSuccess());
        assertEquals(sections, resume.getSections());
        assertEquals(metadata, resume.getMetadata());
    }

    @Test
    @DisplayName("Should set and get Resume properties")
    void settersAndGetters() {
        Resume resume = new Resume();
        Resume.ResumeSection sections = Resume.ResumeSection.builder().build();
        Resume.Metadata metadata = Resume.Metadata.builder().build();

        resume.setSuccess(true);
        resume.setSections(sections);
        resume.setMetadata(metadata);

        assertTrue(resume.isSuccess());
        assertEquals(sections, resume.getSections());
        assertEquals(metadata, resume.getMetadata());
    }

    @Test
    @DisplayName("Should create Education using builder")
    void education_Builder() {
        Resume.Education education = Resume.Education.builder()
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
        Resume.Experience experience = Resume.Experience.builder()
                .title("Software Engineer")
                .company("Google")
                .date("01/2020 - Present")
                .isPresent(true)
                .location("Mountain View, CA")
                .responsibilities(Arrays.asList("Develop features", "Code reviews"))
                .build();

        assertEquals("Software Engineer", experience.getTitle());
        assertEquals("Google", experience.getCompany());
        assertEquals("01/2020 - Present", experience.getDate());
        assertTrue(experience.getIsPresent());
        assertEquals("Mountain View, CA", experience.getLocation());
        assertEquals(2, experience.getResponsibilities().size());
    }

    @Test
    @DisplayName("Should create Project using builder")
    void project_Builder() {
        Resume.Project project = Resume.Project.builder()
                .name("Resume Parser")
                .url("https://github.com/example/resume-parser")
                .description(Arrays.asList("AI-powered parsing", "Multi-format support"))
                .technologies("Java, Spring Boot, ML")
                .date("2023")
                .build();

        assertEquals("Resume Parser", project.getName());
        assertEquals("https://github.com/example/resume-parser", project.getUrl());
        assertEquals(2, project.getDescription().size());
        assertEquals("Java, Spring Boot, ML", project.getTechnologies());
        assertEquals("2023", project.getDate());
    }

    @Test
    @DisplayName("Should create Publication using builder")
    void publication_Builder() {
        Resume.Publication publication = Resume.Publication.builder()
                .title("Machine Learning Paper")
                .year("2023")
                .doi("10.1234/example.doi")
                .build();

        assertEquals("Machine Learning Paper", publication.getTitle());
        assertEquals("2023", publication.getYear());
        assertEquals("10.1234/example.doi", publication.getDoi());
    }

    @Test
    @DisplayName("Should create Contact using builder")
    void contact_Builder() {
        Resume.Contact contact = Resume.Contact.builder()
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
        Resume.Metadata metadata = Resume.Metadata.builder()
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
    @DisplayName("Should create ResumeSection using builder with all sections")
    void resumeSection_Builder() {
        Resume.ResumeSection section = Resume.ResumeSection.builder()
                .education(Collections.singletonList(Resume.Education.builder().degree("BS").build()))
                .experience(Collections.singletonList(Resume.Experience.builder().title("Dev").build()))
                .project(Collections.singletonList(Resume.Project.builder().name("Proj").build()))
                .publication(Collections.singletonList(Resume.Publication.builder().title("Paper").build()))
                .contact(Resume.Contact.builder().name("John").build())
                .build();

        assertEquals(1, section.getEducation().size());
        assertEquals(1, section.getExperience().size());
        assertEquals(1, section.getProject().size());
        assertEquals(1, section.getPublication().size());
        assertEquals("John", section.getContact().getName());
    }

    @Test
    @DisplayName("Should handle null sections in ResumeSection")
    void resumeSection_NullSections() {
        Resume.ResumeSection section = Resume.ResumeSection.builder()
                .education(null)
                .experience(null)
                .project(null)
                .publication(null)
                .contact(null)
                .build();

        assertNull(section.getEducation());
        assertNull(section.getExperience());
        assertNull(section.getProject());
        assertNull(section.getPublication());
        assertNull(section.getContact());
    }

    @Test
    @DisplayName("Should use setters in nested classes")
    void nestedClasses_Setters() {
        Resume.Education education = new Resume.Education();
        education.setDegree("MS");
        education.setMajor("Data Science");
        assertEquals("MS", education.getDegree());
        assertEquals("Data Science", education.getMajor());

        Resume.Experience experience = new Resume.Experience();
        experience.setTitle("Manager");
        experience.setIsPresent(false);
        assertEquals("Manager", experience.getTitle());
        assertFalse(experience.getIsPresent());

        Resume.Contact contact = new Resume.Contact();
        contact.setName("Jane");
        contact.setEmail("jane@test.com");
        assertEquals("Jane", contact.getName());
        assertEquals("jane@test.com", contact.getEmail());
    }

    @Test
    @DisplayName("Should handle empty lists")
    void emptyLists() {
        Resume.ResumeSection section = Resume.ResumeSection.builder()
                .education(Collections.emptyList())
                .experience(Collections.emptyList())
                .project(Collections.emptyList())
                .publication(Collections.emptyList())
                .build();

        assertTrue(section.getEducation().isEmpty());
        assertTrue(section.getExperience().isEmpty());
        assertTrue(section.getProject().isEmpty());
        assertTrue(section.getPublication().isEmpty());
    }

    @Test
    @DisplayName("Should test equals and hashCode")
    void equalsAndHashCode() {
        Resume.Contact contact1 = Resume.Contact.builder().name("John").email("john@test.com").build();
        Resume.Contact contact2 = Resume.Contact.builder().name("John").email("john@test.com").build();

        assertEquals(contact1, contact2);
        assertEquals(contact1.hashCode(), contact2.hashCode());
    }

    @Test
    @DisplayName("Should test toString")
    void toStringTest() {
        Resume.Contact contact = Resume.Contact.builder().name("John").build();

        String str = contact.toString();
        assertTrue(str.contains("John"));
    }
}
