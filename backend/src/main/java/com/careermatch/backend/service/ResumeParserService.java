package com.careermatch.backend.service;

import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.parse.ResumeParserFactory;
import com.careermatch.backend.model.Resume;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/**
 * Service for parsing resume files
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ResumeParserService {
    
    private final ResumeParserFactory resumeParserFactory;

    /**
     * Parse the resume file using the specified strategy
     * @param file the resume file
     * @param strategyType the parsing strategy type (AI or ML)
     * @return parsed resume data
     */
    public Resume parseResume(MultipartFile file, ParsingStrategyType strategyType) {
        return resumeParserFactory.parse(file, strategyType);
    }
}