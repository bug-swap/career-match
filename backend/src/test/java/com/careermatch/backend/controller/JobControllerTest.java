package com.careermatch.backend.controller;

import com.careermatch.backend.dto.response.JobWithScore;
import com.careermatch.backend.dto.response.SimilarJobsResponse;
import com.careermatch.backend.service.JobService;
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

import java.math.BigDecimal;
import java.time.OffsetDateTime;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class JobControllerTest {

    @Mock
    private JobService jobService;

    @InjectMocks
    private JobController jobController;

    private MultipartFile mockResumeFile;
    private SimilarJobsResponse mockResponse;

    @BeforeEach
    void setUp() {
        mockResumeFile = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test resume content".getBytes());

        List<JobWithScore> jobs = Arrays.asList(
                JobWithScore.builder()
                        .id("1")
                        .title("Software Engineer")
                        .company("Tech Corp")
                        .location("San Francisco, CA")
                        .jobType("Full-time")
                        .isRemote(true)
                        .minAmount(new BigDecimal("100000"))
                        .maxAmount(new BigDecimal("150000"))
                        .currency("USD")
                        .jobUrl("https://example.com/job/1")
                        .category("Engineering")
                        .score(0.95)
                        .datePosted(OffsetDateTime.now())
                        .build());

        mockResponse = SimilarJobsResponse.builder()
                .success(true)
                .category("Engineering")
                .jobs(jobs)
                .count(1)
                .message("Successfully found 1 similar jobs")
                .build();
    }

    @Test
    @DisplayName("Should return similar jobs successfully")
    void getSimilarJobsByResume_Success() {
        when(jobService.getSimilarJobsByResume(any(MultipartFile.class), isNull(), eq(10)))
                .thenReturn(mockResponse);

        ResponseEntity<SimilarJobsResponse> response = jobController.getSimilarJobsByResume(
                mockResumeFile, null, 10);

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());
        assertNotNull(response.getBody());
        assertTrue(response.getBody().getSuccess());
        assertEquals(1, response.getBody().getCount());
        assertEquals("Engineering", response.getBody().getCategory());

        verify(jobService, times(1)).getSimilarJobsByResume(any(MultipartFile.class), isNull(), eq(10));
    }

    @Test
    @DisplayName("Should return similar jobs with category override")
    void getSimilarJobsByResume_WithCategoryOverride() {
        when(jobService.getSimilarJobsByResume(any(MultipartFile.class), eq("SALES"), eq(5)))
                .thenReturn(mockResponse);

        ResponseEntity<SimilarJobsResponse> response = jobController.getSimilarJobsByResume(
                mockResumeFile, "SALES", 5);

        assertNotNull(response);
        assertEquals(HttpStatus.OK, response.getStatusCode());

        verify(jobService, times(1)).getSimilarJobsByResume(any(MultipartFile.class), eq("SALES"), eq(5));
    }

    @Test
    @DisplayName("Should return bad request when IllegalStateException is thrown")
    void getSimilarJobsByResume_IllegalStateException() {
        when(jobService.getSimilarJobsByResume(any(MultipartFile.class), any(), anyInt()))
                .thenThrow(new IllegalStateException("Resume processing failed"));

        ResponseEntity<SimilarJobsResponse> response = jobController.getSimilarJobsByResume(
                mockResumeFile, null, 10);

        assertNotNull(response);
        assertEquals(HttpStatus.BAD_REQUEST, response.getStatusCode());
        assertNull(response.getBody());
    }

    @Test
    @DisplayName("Should return internal server error on generic exception")
    void getSimilarJobsByResume_GenericException() {
        when(jobService.getSimilarJobsByResume(any(MultipartFile.class), any(), anyInt()))
                .thenThrow(new RuntimeException("Unexpected error"));

        ResponseEntity<SimilarJobsResponse> response = jobController.getSimilarJobsByResume(
                mockResumeFile, null, 10);

        assertNotNull(response);
        assertEquals(HttpStatus.INTERNAL_SERVER_ERROR, response.getStatusCode());
        assertNull(response.getBody());
    }

    @Test
    @DisplayName("Should use default limit when not provided")
    void getSimilarJobsByResume_DefaultLimit() {
        when(jobService.getSimilarJobsByResume(any(MultipartFile.class), isNull(), eq(10)))
                .thenReturn(mockResponse);

        ResponseEntity<SimilarJobsResponse> response = jobController.getSimilarJobsByResume(
                mockResumeFile, null, 10);

        assertEquals(HttpStatus.OK, response.getStatusCode());
        verify(jobService).getSimilarJobsByResume(any(), isNull(), eq(10));
    }
}
