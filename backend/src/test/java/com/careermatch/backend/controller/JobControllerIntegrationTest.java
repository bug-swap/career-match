package com.careermatch.backend.controller;

import com.careermatch.backend.dto.response.*;
import com.careermatch.backend.service.JobService;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import java.util.Arrays;
import java.util.List;

import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(JobController.class)
class JobControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private JobService jobService;

    @Test
    @DisplayName("POST /api/v1/jobs/match - Should return similar jobs")
    void getSimilarJobsByResume_Success() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        List<JobWithScore> jobs = Arrays.asList(
                JobWithScore.builder()
                        .id("1")
                        .title("Software Engineer")
                        .company("Tech Corp")
                        .score(0.95)
                        .build());

        SimilarJobsResponse response = SimilarJobsResponse.builder()
                .success(true)
                .category("ENGINEERING")
                .jobs(jobs)
                .count(1)
                .message("Found 1 jobs")
                .build();

        when(jobService.getSimilarJobsByResume(any(), isNull(), eq(10)))
                .thenReturn(response);

        mockMvc.perform(multipart("/api/v1/jobs/match")
                .file(file))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success").value(true))
                .andExpect(jsonPath("$.category").value("ENGINEERING"))
                .andExpect(jsonPath("$.count").value(1))
                .andExpect(jsonPath("$.jobs[0].title").value("Software Engineer"));
    }

    @Test
    @DisplayName("POST /api/v1/jobs/match - Should accept category override")
    void getSimilarJobsByResume_WithCategoryOverride() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        SimilarJobsResponse response = SimilarJobsResponse.builder()
                .success(true)
                .category("SALES")
                .jobs(Arrays.asList())
                .count(0)
                .build();

        when(jobService.getSimilarJobsByResume(any(), eq("SALES"), eq(5)))
                .thenReturn(response);

        mockMvc.perform(multipart("/api/v1/jobs/match")
                .file(file)
                .param("category", "SALES")
                .param("limit", "5"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.category").value("SALES"));
    }

    @Test
    @DisplayName("POST /api/v1/jobs/match - Should return 400 on IllegalStateException")
    void getSimilarJobsByResume_BadRequest() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(jobService.getSimilarJobsByResume(any(), any(), anyInt()))
                .thenThrow(new IllegalStateException("Failed to process resume"));

        mockMvc.perform(multipart("/api/v1/jobs/match")
                .file(file))
                .andExpect(status().isBadRequest());
    }

    @Test
    @DisplayName("POST /api/v1/jobs/match - Should return 500 on generic exception")
    void getSimilarJobsByResume_InternalError() throws Exception {
        MockMultipartFile file = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test content".getBytes());

        when(jobService.getSimilarJobsByResume(any(), any(), anyInt()))
                .thenThrow(new RuntimeException("Unexpected error"));

        mockMvc.perform(multipart("/api/v1/jobs/match")
                .file(file))
                .andExpect(status().isInternalServerError());
    }
}
