package com.careermatch.backend.parse.impl;

import com.careermatch.backend.dto.response.SectionsResponse;
import com.careermatch.backend.enums.ParsingStrategyType;
import com.careermatch.backend.model.Resume;
import com.careermatch.backend.service.PythonMLService;
import com.careermatch.backend.parse.ResumeParsingStrategy;
import com.careermatch.backend.parse.ResumeParserFactory;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.stream.Collectors;

/**
 * ML-based parsing strategy using Python ML service
 */
@Slf4j
@Component("mlParsingStrategy")
@RequiredArgsConstructor
public class MLParsingStrategy implements ResumeParsingStrategy {
    
    private final PythonMLService pythonMLService;
    
    @Override
    public Resume parse(MultipartFile file) throws Exception {
        log.info("Using ML parsing strategy for: {}", file.getOriginalFilename());
        long startTime = System.currentTimeMillis();
        
        // Extract text from file for metadata
        String resumeText = ResumeParserFactory.extractTextFromFile(file);

        // Call Python ML service to extract sections
        log.info("Calling Python ML service for section extraction...");
        SectionsResponse sectionsResponse = pythonMLService.extractSections(file);
        
        // Convert SectionsResponse to Resume model
        Resume resume = convertToResume(sectionsResponse);
        
        // Calculate and set metadata
        long processingTime = System.currentTimeMillis() - startTime;
        enrichMetadata(resume, resumeText.length(), processingTime);
        
        log.info("ML parsing completed successfully");
        return resume;
    }
    
    @Override
    public String getStrategyName() {
        return ParsingStrategyType.ML.name();
    }
    
    private Resume convertToResume(SectionsResponse response) {
        Resume.ResumeSection sections = new Resume.ResumeSection();
        
        // Convert education
        if (response.getSections() != null && response.getSections().getEducation() != null) {
            sections.setEducation(response.getSections().getEducation().stream()
                .map(edu -> Resume.Education.builder()
                    .degree(edu.getDegree())
                    .major(edu.getMajor())
                    .institution(edu.getInstitution())
                    .location(edu.getLocation())
                    .date(edu.getDate())
                    .gpa(edu.getGpa())
                    .build())
                .collect(Collectors.toList()));
        } else {
            sections.setEducation(new ArrayList<>());
        }
        
        // Convert experience
        if (response.getSections() != null && response.getSections().getExperience() != null) {
            sections.setExperience(response.getSections().getExperience().stream()
                .map(exp -> Resume.Experience.builder()
                    .title(exp.getTitle())
                    .company(exp.getCompany())
                    .date(exp.getDate())
                    .isPresent(exp.getIsPresent())
                    .location(exp.getLocation())
                    .responsibilities(exp.getResponsibilities() != null ? 
                        exp.getResponsibilities() : new ArrayList<>())
                    .build())
                .collect(Collectors.toList()));
        } else {
            sections.setExperience(new ArrayList<>());
        }
        
        // Convert projects
        if (response.getSections() != null && response.getSections().getProject() != null) {
            sections.setProject(response.getSections().getProject().stream()
                .map(proj -> Resume.Project.builder()
                    .name(proj.getName())
                    .url(proj.getUrl())
                    .description(proj.getDescription() != null ? 
                        proj.getDescription() : new ArrayList<>())
                    .technologies(proj.getTechnologies())
                    .date(proj.getDate())
                    .build())
                .collect(Collectors.toList()));
        } else {
            sections.setProject(new ArrayList<>());
        }
        
        // Convert publications
        if (response.getSections() != null && response.getSections().getPublication() != null) {
            sections.setPublication(response.getSections().getPublication().stream()
                .map(pub -> Resume.Publication.builder()
                    .title(pub.getTitle())
                    .year(pub.getYear())
                    .doi(pub.getDoi())
                    .build())
                .collect(Collectors.toList()));
        } else {
            sections.setPublication(new ArrayList<>());
        }
        
        // Convert contact
        if (response.getSections() != null && response.getSections().getContact() != null) {
            sections.setContact(Resume.Contact.builder()
                .name(response.getSections().getContact().getName())
                .email(response.getSections().getContact().getEmail())
                .phone(response.getSections().getContact().getPhone())
                .linkedin(response.getSections().getContact().getLinkedin())
                .github(response.getSections().getContact().getGithub())
                .website(response.getSections().getContact().getWebsite())
                .build());
        } else {
            sections.setContact(new Resume.Contact());
        }
        
        return Resume.builder()
            .success(response.isSuccess())
            .sections(sections)
            .build();
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
}

