package com.careermatch.backend.controller;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.ResumeParserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * REST Controller for resume parsing operations
 */
@Slf4j
@RestController
@RequestMapping("/api/v1/resume")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ResumeController {

    private final ResumeParserService resumeParserService;

    /**
     * Upload and parse a resume file with default strategy
     * @param file the resume file
     * @return parsed resume data
     */
    @PostMapping("/parse")
    public ResponseEntity<?> parseResume(@RequestParam("file") MultipartFile file) {
        return parseResumeWithStrategy(file, null);
    }

    /**
     * Upload and parse a resume file with specified strategy
     * @param file the resume file
     * @param strategy parsing strategy (AI or ML)
     * @return parsed resume data
     */
    @PostMapping("/parse/{strategy}")
    public ResponseEntity<?> parseResumeWithStrategy(
            @RequestParam("file") MultipartFile file,
            @PathVariable(required = false) String strategy) {
        try {
            log.info("Received file upload request: {} with strategy: {}",
                    file.getOriginalFilename(), strategy != null ? strategy : "default");

            ParsingStrategyType strategyType = ParsingStrategyType.fromString(strategy);
            Resume resume = resumeParserService.parseResume(file, strategyType);

            return ResponseEntity.ok(resume);

        } catch (IllegalArgumentException e) {
            log.error("Validation error: {}", e.getMessage());
            return ResponseEntity.badRequest().body(
                    new ErrorResponse("Validation Error", e.getMessage()));

        } catch (Exception e) {
            log.error("Error parsing resume", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(
                    new ErrorResponse("Parsing Error", "Failed to parse resume: " + e.getMessage()));
        }
    }

    /**
     * Health check endpoint
     */
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("Resume Parser Service is running");
    }

    // Response DTOs
    record ErrorResponse(String error, String message) {}
}