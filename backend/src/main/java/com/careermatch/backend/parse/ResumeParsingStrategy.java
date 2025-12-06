package com.careermatch.backend.parse;

import com.careermatch.backend.model.Resume;
import org.springframework.web.multipart.MultipartFile;

/**
 * Strategy interface for resume parsing
 */
public interface ResumeParsingStrategy {
    /**
     * Parse resume using specific strategy
     * @param file resume file to parse
     * @return parsed Resume object
     * @throws Exception if parsing fails
     */
    Resume parse(MultipartFile file) throws Exception;

    /**
     * Get strategy name
     * @return name of the strategy
     */
    String getStrategyName();
}