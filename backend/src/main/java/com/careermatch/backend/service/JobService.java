package com.careermatch.backend.service;

import com.careermatch.backend.dto.request.JobFilterRequest;
import com.careermatch.backend.dto.response.*;
import com.careermatch.backend.entity.Job;
import com.careermatch.backend.repository.JobRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
public class JobService {

    private final JobRepository jobRepository;
    private final PythonMLService pythonMLService;
    private final RpcService rpcService;

    /**
     * Get similar jobs by resume using embeddings and category classification
     * @param resumeFile The resume file to analyze
     * @param categoryOverride Optional category to override ML classification
     * @param limit Maximum number of results to return (default: 10)
     * @return Similar jobs response from RPC
     */
    public SimilarJobsResponse getSimilarJobsByResume(MultipartFile resumeFile, String categoryOverride, Integer limit) {
        log.info("Processing resume to find similar jobs: {}", resumeFile.getOriginalFilename());

        // Get embeddings from Python ML Service
        EmbeddingResponse embeddingResponse = pythonMLService.generateEmbedding(resumeFile);
        if (embeddingResponse == null || !embeddingResponse.getSuccess() || embeddingResponse.getEmbedding() == null) {
            throw new IllegalStateException("Failed to generate embeddings from resume");
        }

        Double[] embedding = embeddingResponse.getEmbedding();
        log.info("Successfully generated embedding with {} dimensions", embedding.length);

        // Determine which category to use
        String searchCategory;
        Classification classification = null;

        if (categoryOverride != null && !categoryOverride.trim().isEmpty()) {
            // Use provided category override
            searchCategory = categoryOverride.trim().toUpperCase();
            log.info("Using category override: {}", searchCategory);
        } else {
            // Get category classification from Python ML Service
            CategoryResponse categoryResponse = pythonMLService.classifyCategory(resumeFile);
            if (categoryResponse == null || !categoryResponse.getSuccess() || categoryResponse.getClassification() == null) {
                throw new IllegalStateException("Failed to classify resume category");
            }

            classification = categoryResponse.getClassification();
            searchCategory = classification.getCategory();
            log.info("Resume classified as category: {} with confidence: {}", searchCategory, classification.getConfidence());
        }

        // Call RPC to get similar jobs
        SimilarJobsResponse response = rpcService.getSimilarJobsByCategory(embedding, searchCategory, limit);

        // Add category classification details to response
        response.setCategory(searchCategory);
        if (classification != null && classification.getTop3() != null) {
            List<SimilarJobsResponse.CategoryInfo> categoryInfoList = classification.getTop3().stream()
                    .map(detail -> SimilarJobsResponse.CategoryInfo.builder()
                            .category(detail.getCategory())
                            .confidence((int) Math.round(detail.getConfidence() * 100))
                            .build())
                    .collect(java.util.stream.Collectors.toList());
            response.setCategories(categoryInfoList);
        }

        return response;
    }
}