package com.careermatch.backend.dto.request;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;

import static org.junit.jupiter.api.Assertions.*;

class JobFilterRequestTest {

    @Test
    @DisplayName("Should create JobFilterRequest with default values")
    void defaultValues() {
        JobFilterRequest request = new JobFilterRequest();

        assertNull(request.getCategory());
        assertNull(request.getLocation());
        assertNull(request.getJobType());
        assertNull(request.getIsRemote());
        assertNull(request.getMinSalary());
        assertNull(request.getMaxSalary());
        assertNull(request.getSearchQuery());
        assertEquals(0, request.getPage());
        assertEquals(20, request.getSize());
        assertEquals("datePosted", request.getSortBy());
        assertEquals("DESC", request.getSortOrder());
    }

    @Test
    @DisplayName("Should create JobFilterRequest with all-args constructor")
    void allArgsConstructor() {
        JobFilterRequest request = new JobFilterRequest(
                "Engineering",
                "San Francisco",
                "Full-time",
                true,
                new BigDecimal("100000"),
                new BigDecimal("200000"),
                "java developer",
                2,
                50,
                "salary",
                "ASC");

        assertEquals("Engineering", request.getCategory());
        assertEquals("San Francisco", request.getLocation());
        assertEquals("Full-time", request.getJobType());
        assertTrue(request.getIsRemote());
        assertEquals(new BigDecimal("100000"), request.getMinSalary());
        assertEquals(new BigDecimal("200000"), request.getMaxSalary());
        assertEquals("java developer", request.getSearchQuery());
        assertEquals(2, request.getPage());
        assertEquals(50, request.getSize());
        assertEquals("salary", request.getSortBy());
        assertEquals("ASC", request.getSortOrder());
    }

    @Test
    @DisplayName("Should set and get all properties")
    void settersAndGetters() {
        JobFilterRequest request = new JobFilterRequest();

        request.setCategory("Data Science");
        request.setLocation("New York");
        request.setJobType("Contract");
        request.setIsRemote(false);
        request.setMinSalary(new BigDecimal("80000"));
        request.setMaxSalary(new BigDecimal("120000"));
        request.setSearchQuery("python");
        request.setPage(1);
        request.setSize(25);
        request.setSortBy("company");
        request.setSortOrder("DESC");

        assertEquals("Data Science", request.getCategory());
        assertEquals("New York", request.getLocation());
        assertEquals("Contract", request.getJobType());
        assertFalse(request.getIsRemote());
        assertEquals(new BigDecimal("80000"), request.getMinSalary());
        assertEquals(new BigDecimal("120000"), request.getMaxSalary());
        assertEquals("python", request.getSearchQuery());
        assertEquals(1, request.getPage());
        assertEquals(25, request.getSize());
        assertEquals("company", request.getSortBy());
        assertEquals("DESC", request.getSortOrder());
    }

    @Test
    @DisplayName("Should handle null values for optional fields")
    void nullOptionalFields() {
        JobFilterRequest request = new JobFilterRequest();

        request.setCategory(null);
        request.setLocation(null);
        request.setJobType(null);
        request.setIsRemote(null);
        request.setMinSalary(null);
        request.setMaxSalary(null);
        request.setSearchQuery(null);

        assertNull(request.getCategory());
        assertNull(request.getLocation());
        assertNull(request.getJobType());
        assertNull(request.getIsRemote());
        assertNull(request.getMinSalary());
        assertNull(request.getMaxSalary());
        assertNull(request.getSearchQuery());
    }

    @Test
    @DisplayName("Should test equals and hashCode")
    void equalsAndHashCode() {
        JobFilterRequest request1 = new JobFilterRequest();
        request1.setCategory("Engineering");
        request1.setPage(1);

        JobFilterRequest request2 = new JobFilterRequest();
        request2.setCategory("Engineering");
        request2.setPage(1);

        assertEquals(request1, request2);
        assertEquals(request1.hashCode(), request2.hashCode());
    }

    @Test
    @DisplayName("Should test toString")
    void toStringTest() {
        JobFilterRequest request = new JobFilterRequest();
        request.setCategory("Tech");

        String str = request.toString();
        assertTrue(str.contains("Tech"));
    }
}
