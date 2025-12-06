package com.careermatch.backend.controller;

import com.careermatch.backend.dto.response.SimilarJobsResponse;
import com.careermatch.backend.service.JobService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@Slf4j
@RestController
@RequestMapping("/api/v1/jobs")
@RequiredArgsConstructor
public class JobController {

    private final JobService jobService;

    /**
     * Get similar jobs based on resume analysis
     * Extracts embeddings and category from resume, then calls RPC to find similar jobs
     * @param resumeFile The resume file to analyze
     * @param category Optional category to override ML classification
     * @param limit Maximum number of results to return (optional, default: 10)
     * @return Similar jobs found via RPC
     */
    @PostMapping(value = "/match", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<SimilarJobsResponse> getSimilarJobsByResume(
            @RequestParam("file") MultipartFile resumeFile,
            @RequestParam(value = "category", required = false) String category,
            @RequestParam(value = "limit", required = false, defaultValue = "10") Integer limit) {
        log.info("Received request to find similar jobs for resume: {} (category: {}, limit: {})",
                resumeFile.getOriginalFilename(), category != null ? category : "auto", limit);

        try {
            SimilarJobsResponse response = jobService.getSimilarJobsByResume(resumeFile, category, limit);
            return ResponseEntity.ok(response);
        } catch (IllegalStateException e) {
            log.error("Resume processing failed: {}", e.getMessage());
            return ResponseEntity.badRequest().build();
        } catch (Exception e) {
            log.error("Failed to find similar jobs: {}", e.getMessage(), e);
            return ResponseEntity.internalServerError().build();
        }
    }
}
