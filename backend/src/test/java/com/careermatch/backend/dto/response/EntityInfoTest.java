package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.Test;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class EntityInfoTest {

    @Test
    void testBuilder() {
        EntityInfo info = EntityInfo.builder()
                .jobTitles(Arrays.asList("Engineer"))
                .companies(Arrays.asList("Google"))
                .workDates(Arrays.asList("2020-2021"))
                .skills(Arrays.asList("Java"))
                .degrees(Arrays.asList("BS"))
                .majors(Arrays.asList("CS"))
                .institutions(Arrays.asList("MIT"))
                .graduationYears(Arrays.asList("2020"))
                .gpa("3.8")
                .certifications(Arrays.asList("AWS"))
                .projects(Arrays.asList("Project1"))
                .publications(Arrays.asList("Paper1"))
                .languages(Arrays.asList("English"))
                .summary("Summary")
                .build();

        assertEquals(1, info.getJobTitles().size());
        assertEquals("Engineer", info.getJobTitles().get(0));
        assertEquals("Google", info.getCompanies().get(0));
        assertEquals("2020-2021", info.getWorkDates().get(0));
        assertEquals("Java", info.getSkills().get(0));
        assertEquals("BS", info.getDegrees().get(0));
        assertEquals("CS", info.getMajors().get(0));
        assertEquals("MIT", info.getInstitutions().get(0));
        assertEquals("2020", info.getGraduationYears().get(0));
        assertEquals("3.8", info.getGpa());
        assertEquals("AWS", info.getCertifications().get(0));
        assertEquals("Project1", info.getProjects().get(0));
        assertEquals("Paper1", info.getPublications().get(0));
        assertEquals("English", info.getLanguages().get(0));
        assertEquals("Summary", info.getSummary());
    }

    @Test
    void testNoArgsAndSetters() {
        EntityInfo info = new EntityInfo();
        List<String> list = Arrays.asList("test");

        info.setJobTitles(list);
        info.setCompanies(list);
        info.setWorkDates(list);
        info.setSkills(list);
        info.setDegrees(list);
        info.setMajors(list);
        info.setInstitutions(list);
        info.setGraduationYears(list);
        info.setGpa("4.0");
        info.setCertifications(list);
        info.setProjects(list);
        info.setPublications(list);
        info.setLanguages(list);
        info.setSummary("text");

        assertEquals(list, info.getJobTitles());
        assertEquals(list, info.getCompanies());
        assertEquals(list, info.getWorkDates());
        assertEquals(list, info.getSkills());
        assertEquals(list, info.getDegrees());
        assertEquals(list, info.getMajors());
        assertEquals(list, info.getInstitutions());
        assertEquals(list, info.getGraduationYears());
        assertEquals("4.0", info.getGpa());
        assertEquals(list, info.getCertifications());
        assertEquals(list, info.getProjects());
        assertEquals(list, info.getPublications());
        assertEquals(list, info.getLanguages());
        assertEquals("text", info.getSummary());
    }

    @Test
    void testAllArgsConstructor() {
        List<String> list = Arrays.asList("a");
        EntityInfo info = new EntityInfo(list, list, list, list, list, list, list, list, "3.5", list, list, list, list,
                "sum");

        assertEquals(list, info.getJobTitles());
        assertEquals("3.5", info.getGpa());
        assertEquals("sum", info.getSummary());
    }
}
