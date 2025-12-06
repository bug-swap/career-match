package com.careermatch.backend.dto.response;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class SimilarJobsResponseTest {

    @Test
    @DisplayName("Should create SimilarJobsResponse using builder")
    void builder() {
        List<JobWithScore> jobs = Collections.singletonList(
                JobWithScore.builder().id("1").title("Engineer").build());
        List<SimilarJobsResponse.CategoryInfo> categories = Arrays.asList(
                SimilarJobsResponse.CategoryInfo.builder().category("Tech").confidence(95).build());

        SimilarJobsResponse response = SimilarJobsResponse.builder()
                .success(true)
                .category("Engineering")
                .categories(categories)
                .jobs(jobs)
                .message("Found jobs")
                .count(1)
                .build();

        assertTrue(response.getSuccess());
        assertEquals("Engineering", response.getCategory());
        assertEquals(1, response.getCategories().size());
        assertEquals(1, response.getJobs().size());
        assertEquals("Found jobs", response.getMessage());
        assertEquals(1, response.getCount());
    }

    @Test
    @DisplayName("Should create SimilarJobsResponse using no-args constructor")
    void noArgsConstructor() {
        SimilarJobsResponse response = new SimilarJobsResponse();

        assertNull(response.getSuccess());
        assertNull(response.getCategory());
        assertNull(response.getCategories());
        assertNull(response.getJobs());
    }

    @Test
    @DisplayName("Should set and get all properties")
    void settersAndGetters() {
        SimilarJobsResponse response = new SimilarJobsResponse();

        response.setSuccess(true);
        response.setCategory("Sales");
        response.setCategories(Collections.emptyList());
        response.setJobs(Collections.emptyList());
        response.setMessage("Test message");
        response.setCount(5);

        assertTrue(response.getSuccess());
        assertEquals("Sales", response.getCategory());
        assertTrue(response.getCategories().isEmpty());
        assertTrue(response.getJobs().isEmpty());
        assertEquals("Test message", response.getMessage());
        assertEquals(5, response.getCount());
    }

    @Test
    @DisplayName("Should create CategoryInfo using builder")
    void categoryInfo_Builder() {
        SimilarJobsResponse.CategoryInfo info = SimilarJobsResponse.CategoryInfo.builder()
                .category("Engineering")
                .confidence(95)
                .build();

        assertEquals("Engineering", info.getCategory());
        assertEquals(95, info.getConfidence());
    }

    @Test
    @DisplayName("Should test equals and hashCode")
    void equalsAndHashCode() {
        SimilarJobsResponse response1 = SimilarJobsResponse.builder()
                .success(true)
                .category("Tech")
                .count(10)
                .build();

        SimilarJobsResponse response2 = SimilarJobsResponse.builder()
                .success(true)
                .category("Tech")
                .count(10)
                .build();

        assertEquals(response1, response2);
        assertEquals(response1.hashCode(), response2.hashCode());
    }
}

class JobWithScoreTest {

    @Test
    @DisplayName("Should create JobWithScore using builder")
    void builder() {
        OffsetDateTime now = OffsetDateTime.now();

        JobWithScore job = JobWithScore.builder()
                .id("123")
                .title("Software Engineer")
                .company("Tech Corp")
                .location("San Francisco")
                .datePosted(now)
                .jobType("Full-time")
                .isRemote(true)
                .minAmount(new BigDecimal("100000"))
                .maxAmount(new BigDecimal("150000"))
                .currency("USD")
                .jobUrl("https://example.com")
                .category("Engineering")
                .score(0.95)
                .build();

        assertEquals("123", job.getId());
        assertEquals("Software Engineer", job.getTitle());
        assertEquals("Tech Corp", job.getCompany());
        assertEquals("San Francisco", job.getLocation());
        assertEquals(now, job.getDatePosted());
        assertEquals("Full-time", job.getJobType());
        assertTrue(job.getIsRemote());
        assertEquals(new BigDecimal("100000"), job.getMinAmount());
        assertEquals(new BigDecimal("150000"), job.getMaxAmount());
        assertEquals("USD", job.getCurrency());
        assertEquals("https://example.com", job.getJobUrl());
        assertEquals("Engineering", job.getCategory());
        assertEquals(0.95, job.getScore());
    }

    @Test
    @DisplayName("Should create JobWithScore using no-args constructor")
    void noArgsConstructor() {
        JobWithScore job = new JobWithScore();

        assertNull(job.getId());
        assertNull(job.getTitle());
        assertNull(job.getScore());
    }

    @Test
    @DisplayName("Should set and get all properties")
    void settersAndGetters() {
        JobWithScore job = new JobWithScore();

        job.setId("456");
        job.setTitle("Data Scientist");
        job.setScore(0.88);

        assertEquals("456", job.getId());
        assertEquals("Data Scientist", job.getTitle());
        assertEquals(0.88, job.getScore());
    }
}

class EmbeddingResponseTest {

    @Test
    @DisplayName("Should create EmbeddingResponse using builder")
    void builder() {
        Double[] embedding = { 0.1, 0.2, 0.3, 0.4, 0.5 };

        EmbeddingResponse response = EmbeddingResponse.builder()
                .success(true)
                .embedding(embedding)
                .build();

        assertTrue(response.getSuccess());
        assertArrayEquals(embedding, response.getEmbedding());
    }

    @Test
    @DisplayName("Should create EmbeddingResponse using no-args constructor")
    void noArgsConstructor() {
        EmbeddingResponse response = new EmbeddingResponse();

        assertNull(response.getSuccess());
        assertNull(response.getEmbedding());
    }

    @Test
    @DisplayName("Should handle null embedding")
    void nullEmbedding() {
        EmbeddingResponse response = EmbeddingResponse.builder()
                .success(true)
                .embedding(null)
                .build();

        assertTrue(response.getSuccess());
        assertNull(response.getEmbedding());
    }

    @Test
    @DisplayName("Should handle large embedding array")
    void largeEmbedding() {
        Double[] largeEmbedding = new Double[1536];
        for (int i = 0; i < 1536; i++) {
            largeEmbedding[i] = Math.random();
        }

        EmbeddingResponse response = EmbeddingResponse.builder()
                .success(true)
                .embedding(largeEmbedding)
                .build();

        assertEquals(1536, response.getEmbedding().length);
    }
}

class CategoryResponseTest {

    @Test
    @DisplayName("Should create CategoryResponse using builder")
    void builder() {
        Classification classification = Classification.builder()
                .category("Engineering")
                .confidence(0.95)
                .build();

        CategoryResponse response = CategoryResponse.builder()
                .success(true)
                .classification(classification)
                .build();

        assertTrue(response.getSuccess());
        assertNotNull(response.getClassification());
        assertEquals("Engineering", response.getClassification().getCategory());
    }

    @Test
    @DisplayName("Should create CategoryResponse using no-args constructor")
    void noArgsConstructor() {
        CategoryResponse response = new CategoryResponse();

        assertNull(response.getSuccess());
        assertNull(response.getClassification());
    }
}

class ClassificationTest {

    @Test
    @DisplayName("Should create Classification using builder")
    void builder() {
        List<ClassificationDetail> top3 = Arrays.asList(
                ClassificationDetail.builder().category("Engineering").confidence(0.95).build(),
                ClassificationDetail.builder().category("IT").confidence(0.03).build(),
                ClassificationDetail.builder().category("Data").confidence(0.02).build());

        Classification classification = Classification.builder()
                .category("Engineering")
                .confidence(0.95)
                .top3(top3)
                .build();

        assertEquals("Engineering", classification.getCategory());
        assertEquals(0.95, classification.getConfidence());
        assertEquals(3, classification.getTop3().size());
    }

    @Test
    @DisplayName("Should create Classification using no-args constructor")
    void noArgsConstructor() {
        Classification classification = new Classification();

        assertNull(classification.getCategory());
        assertNull(classification.getConfidence());
        assertNull(classification.getTop3());
    }

    @Test
    @DisplayName("Should handle null top3")
    void nullTop3() {
        Classification classification = Classification.builder()
                .category("Sales")
                .confidence(0.90)
                .top3(null)
                .build();

        assertNull(classification.getTop3());
    }
}

class ClassificationDetailTest {

    @Test
    @DisplayName("Should create ClassificationDetail using builder")
    void builder() {
        ClassificationDetail detail = ClassificationDetail.builder()
                .category("Engineering")
                .confidence(0.95)
                .build();

        assertEquals("Engineering", detail.getCategory());
        assertEquals(0.95, detail.getConfidence());
    }

    @Test
    @DisplayName("Should create ClassificationDetail using no-args constructor")
    void noArgsConstructor() {
        ClassificationDetail detail = new ClassificationDetail();

        assertNull(detail.getCategory());
        assertNull(detail.getConfidence());
    }

    @Test
    @DisplayName("Should set and get properties")
    void settersAndGetters() {
        ClassificationDetail detail = new ClassificationDetail();

        detail.setCategory("Marketing");
        detail.setConfidence(0.75);

        assertEquals("Marketing", detail.getCategory());
        assertEquals(0.75, detail.getConfidence());
    }
}
