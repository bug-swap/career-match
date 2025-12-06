package com.careermatch.backend.entity;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.OffsetDateTime;

import static org.junit.jupiter.api.Assertions.*;

class JobTest {

    @Test
    @DisplayName("Should create job using builder")
    void builder() {
        OffsetDateTime now = OffsetDateTime.now();

        Job job = Job.builder()
                .id("123")
                .title("Software Engineer")
                .company("Tech Corp")
                .location("San Francisco, CA")
                .category("Engineering")
                .jobType("Full-time")
                .isRemote(true)
                .minAmount(new BigDecimal("100000"))
                .maxAmount(new BigDecimal("150000"))
                .currency("USD")
                .jobUrl("https://example.com/job/123")
                .datePosted(now)
                .description("Job description here")
                .build();

        assertEquals("123", job.getId());
        assertEquals("Software Engineer", job.getTitle());
        assertEquals("Tech Corp", job.getCompany());
        assertEquals("San Francisco, CA", job.getLocation());
        assertEquals("Engineering", job.getCategory());
        assertEquals("Full-time", job.getJobType());
        assertTrue(job.getIsRemote());
        assertEquals(new BigDecimal("100000"), job.getMinAmount());
        assertEquals(new BigDecimal("150000"), job.getMaxAmount());
        assertEquals("USD", job.getCurrency());
        assertEquals("https://example.com/job/123", job.getJobUrl());
        assertEquals(now, job.getDatePosted());
        assertEquals("Job description here", job.getDescription());
    }

    @Test
    @DisplayName("Should create job using no-args constructor")
    void noArgsConstructor() {
        Job job = new Job();

        assertNull(job.getId());
        assertNull(job.getTitle());
        assertNull(job.getCompany());
    }

    @Test
    @DisplayName("Should create job using all-args constructor")
    void allArgsConstructor() {
        OffsetDateTime now = OffsetDateTime.now();

        Job job = new Job("123", "Engineer", "Company", "Location", "Tech",
                "Full-time", false, new BigDecimal("50000"), new BigDecimal("80000"),
                "EUR", "http://url.com", now, "Description");

        assertEquals("123", job.getId());
        assertEquals("Engineer", job.getTitle());
        assertEquals("Company", job.getCompany());
        assertEquals("Location", job.getLocation());
        assertEquals("Tech", job.getCategory());
        assertEquals("Full-time", job.getJobType());
        assertFalse(job.getIsRemote());
        assertEquals(new BigDecimal("50000"), job.getMinAmount());
        assertEquals(new BigDecimal("80000"), job.getMaxAmount());
        assertEquals("EUR", job.getCurrency());
        assertEquals("http://url.com", job.getJobUrl());
        assertEquals(now, job.getDatePosted());
        assertEquals("Description", job.getDescription());
    }

    @Test
    @DisplayName("Should set and get properties using setters")
    void settersAndGetters() {
        Job job = new Job();
        OffsetDateTime now = OffsetDateTime.now();

        job.setId("456");
        job.setTitle("Data Scientist");
        job.setCompany("Data Inc");
        job.setLocation("New York, NY");
        job.setCategory("Data Science");
        job.setJobType("Part-time");
        job.setIsRemote(true);
        job.setMinAmount(new BigDecimal("120000"));
        job.setMaxAmount(new BigDecimal("180000"));
        job.setCurrency("USD");
        job.setJobUrl("https://example.com/ds");
        job.setDatePosted(now);
        job.setDescription("Data science role");

        assertEquals("456", job.getId());
        assertEquals("Data Scientist", job.getTitle());
        assertEquals("Data Inc", job.getCompany());
        assertEquals("New York, NY", job.getLocation());
        assertEquals("Data Science", job.getCategory());
        assertEquals("Part-time", job.getJobType());
        assertTrue(job.getIsRemote());
        assertEquals(new BigDecimal("120000"), job.getMinAmount());
        assertEquals(new BigDecimal("180000"), job.getMaxAmount());
        assertEquals("USD", job.getCurrency());
        assertEquals("https://example.com/ds", job.getJobUrl());
        assertEquals(now, job.getDatePosted());
        assertEquals("Data science role", job.getDescription());
    }

    @Test
    @DisplayName("Should handle null values")
    void nullValues() {
        Job job = Job.builder()
                .id(null)
                .title(null)
                .company(null)
                .isRemote(null)
                .minAmount(null)
                .maxAmount(null)
                .build();

        assertNull(job.getId());
        assertNull(job.getTitle());
        assertNull(job.getCompany());
        assertNull(job.getIsRemote());
        assertNull(job.getMinAmount());
        assertNull(job.getMaxAmount());
    }

    @Test
    @DisplayName("Should handle long description")
    void longDescription() {
        String longDescription = "A".repeat(10000);
        Job job = Job.builder()
                .description(longDescription)
                .build();

        assertEquals(10000, job.getDescription().length());
    }
}
