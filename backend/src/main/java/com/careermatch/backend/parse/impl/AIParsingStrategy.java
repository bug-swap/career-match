package com.careermatch.backend.parse.impl;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.AiService;
import com.careermatch.backend.parse.ResumeParsingStrategy;
import com.careermatch.backend.parse.ResumeParserFactory;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

/**
 * AI-based parsing strategy using Gemini
 */
@Slf4j
@Component("aiParsingStrategy")
@RequiredArgsConstructor
public class AIParsingStrategy implements ResumeParsingStrategy {
    
    private final AiService aiService;
    private final ObjectMapper objectMapper;
    
    private static final String RESUME_PARSE_PROMPT = """
            You are a resume parsing expert. Parse the provided resume into the exact JSON structure below.
            Extract information carefully and accurately. Return ONLY valid JSON, no additional text.
            
            IMPORTANT PARSING RULES:
            
            CONTACT SECTION:
            - Extract name, email, phone, LinkedIn URL, GitHub URL, and personal website if present
            - Leave empty strings for fields not found in the resume
            
            EDUCATION SECTION:
            - degree: Full degree name (e.g., "Bachelor of Science", "Master of Arts")
            - major: Field of study (e.g., "Computer Science", "Engineering")
            - institution: University or college name
            - location: City, State (e.g., "Boulder, CO")
            - date: Graduation date in format "MM/YYYY" or "YYYY" or date range "MM/YYYY - MM/YYYY"
            - gpa: GPA if listed (e.g., "3.8" or "3.8/4.0"), empty string if not provided
            
            EXPERIENCE SECTION:
            - title: Job title
            - company: Company or organization name
            - date: Employment period in format "MM/YYYY - MM/YYYY" or "MM/YYYY - Present"
            - isPresent: CRITICAL - Set to true ONLY if the date contains "Present", "Current", or no end date is specified. Otherwise always false.
            - location: City, State or remote designation
            - responsibilities: Array of bullet points describing what was done (extract 3-5 key responsibilities)
            
            PROJECT SECTION:
            - name: Project title or name
            - url: GitHub URL or project link if available, empty string otherwise
            - description: Array of bullet points describing the project (1-3 points)
            - technologies: Comma-separated list of tech stack (e.g., "React, Node.js, MongoDB")
            - date: Project completion date or period if available
            
            PUBLICATION SECTION:
            - title: Publication title
            - year: Publication year
            - doi: DOI or publication identifier if available, empty string otherwise
            
            Return this exact JSON structure (fill arrays and objects with extracted data, use empty strings/arrays for missing data):
            ```json
            {
              "success": true,
              "sections": {
                "education": [{"degree": "", "major": "", "institution": "", "location": "", "date": "", "gpa": ""}],
                "experience": [{"title": "", "company": "", "date": "", "isPresent": false, "location": "", "responsibilities": []}],
                "project": [{"name": "", "url": "", "description": [], "technologies": "", "date": ""}],
                "publication": [{"title": "", "year": "", "doi": ""}],
                "contact": {"name": "", "email": "", "phone": "", "linkedin": "", "github": "", "website": ""}
              },
              "metadata": {"section_count": 0, "total_items": 0, "total_char_count": 0, "processing_time_ms": 0}
            }
            ```
            
            Resume to parse:
            %s
            """;
    
    @Override
    public Resume parse(MultipartFile file) throws Exception {
        log.info("Using AI parsing strategy for: {}", file.getOriginalFilename());
        long startTime = System.currentTimeMillis();
        
        // Extract text from file
        String resumeText = ResumeParserFactory.extractTextFromFile(file);
        log.info("Extracted resume text, length: {} characters", resumeText.length());

        // Call AI service using builder pattern with text prompt
        log.info("Calling Gemini AI service for parsing...");
        String aiResponse = aiService.builder()
                .prompt(String.format(RESUME_PARSE_PROMPT, resumeText))
                .maxTokens(1_00_000)
                .temperature(0.3)
                .build()
                .execute();

        // Clean response (remove markdown code blocks if present)
        String cleanedResponse = cleanJsonResponse(aiResponse);
        log.debug("AI Response cleaned, length: {}", cleanedResponse.length());

        // Validate JSON completeness
        validateJsonCompleteness(cleanedResponse);

        // Parse JSON response to Resume object with lenient settings
        Resume resume = parseJsonToResume(cleanedResponse);
        log.info("Parsed resume sections successfully");
        // Calculate and set metadata
        long processingTime = System.currentTimeMillis() - startTime;
        enrichMetadata(resume, resumeText.length(), processingTime);
        log.info("AI parsing completed successfully in {} ms", processingTime);
        return resume;
    }
    
    @Override
    public String getStrategyName() {
        return ParsingStrategyType.AI.name();
    }
    
    private String cleanJsonResponse(String response) {
        String cleaned = response.trim();
        
        // Remove markdown code blocks
        if (cleaned.startsWith("```json")) {
            cleaned = cleaned.substring(7);
        } else if (cleaned.startsWith("```")) {
            cleaned = cleaned.substring(3);
        }
        
        if (cleaned.endsWith("```")) {
            cleaned = cleaned.substring(0, cleaned.length() - 3);
        }
        
        cleaned = cleaned.trim();

        // Replace line breaks within arrays/objects with spaces to maintain valid JSON
        cleaned = cleaned.replaceAll("(?<!\\\\)\\n\\s*(?=[\\w\"-])", " ");

        log.debug("Cleaned JSON response: {}", cleaned.substring(0, Math.min(200, cleaned.length())));

        return cleaned;
    }

    private Resume parseJsonToResume(String json) {
        // Configure ObjectMapper to be lenient
        ObjectMapper lenientMapper = objectMapper.copy();
        lenientMapper.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        lenientMapper.configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, false);
        lenientMapper.configure(DeserializationFeature.ACCEPT_EMPTY_STRING_AS_NULL_OBJECT, true);
        lenientMapper.configure(DeserializationFeature.ACCEPT_EMPTY_ARRAY_AS_NULL_OBJECT, true);

        try {
            return lenientMapper.readValue(json, Resume.class);
        } catch (Exception e) {
            log.error("Failed to parse JSON response, attempting partial recovery", e);
            // If parsing fails completely, create a minimal Resume object
            Resume resume = Resume.builder()
                    .success(false)
                    .sections(Resume.ResumeSection.builder().build())
                    .metadata(Resume.Metadata.builder()
                            .sectionCount(0)
                            .totalItems(0)
                            .totalCharCount(0)
                            .processingTimeMs(0L)
                            .build())
                    .build();
            log.warn("Using fallback Resume object due to parsing error");
            return resume;
        }
    }

    private void enrichMetadata(Resume resume, int charCount, long processingTime) {
        if (resume.getMetadata() == null) {
            resume.setMetadata(new Resume.Metadata());
        }
        
        Resume.Metadata metadata = resume.getMetadata();
        Resume.ResumeSection sections = resume.getSections();
        
        int sectionCount = 0;
        int totalItems = 0;
        
        if (sections != null) {
            if (sections.getEducation() != null && !sections.getEducation().isEmpty()) {
                sectionCount++;
                totalItems += sections.getEducation().size();
            }
            if (sections.getExperience() != null && !sections.getExperience().isEmpty()) {
                sectionCount++;
                totalItems += sections.getExperience().size();
            }
            if (sections.getProject() != null && !sections.getProject().isEmpty()) {
                sectionCount++;
                totalItems += sections.getProject().size();
            }
            if (sections.getPublication() != null && !sections.getPublication().isEmpty()) {
                sectionCount++;
                totalItems += sections.getPublication().size();
            }
            if (sections.getContact() != null) {
                sectionCount++;
            }
        }
        
        metadata.setSectionCount(sectionCount);
        metadata.setTotalItems(totalItems);
        metadata.setTotalCharCount(charCount);
        metadata.setProcessingTimeMs(processingTime);
    }

    private void validateJsonCompleteness(String json) throws Exception {
        String trimmed = json.trim();
        // Basic validation: check if starts with '{' and ends with '}'
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
            throw new Exception("Invalid JSON structure: must start with '{' and end with '}'");
        }

        // Try to find matching braces
        int braceCount = 0;
        for (char c : trimmed.toCharArray()) {
            if (c == '{') braceCount++;
            else if (c == '}') braceCount--;

            if (braceCount < 0) {
                throw new Exception("Invalid JSON: mismatched braces");
            }
        }

        if (braceCount != 0) {
            throw new Exception("Invalid JSON: unmatched braces (count: " + braceCount + ")");
        }
    }
}
