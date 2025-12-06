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
     * @return Similar jobs response from RPC
     */
    public SimilarJobsResponse getSimilarJobsByResume(MultipartFile resumeFile) {
        log.info("Processing resume to find similar jobs: {}", resumeFile.getOriginalFilename());

        // Get embeddings from Python ML Service
        EmbeddingResponse embeddingResponse = pythonMLService.generateEmbedding(resumeFile);
        if (embeddingResponse == null || !embeddingResponse.getSuccess() || embeddingResponse.getEmbedding() == null) {
            throw new IllegalStateException("Failed to generate embeddings from resume");
        }

        Double[] embedding = embeddingResponse.getEmbedding();
        log.info("Successfully generated embedding with {} dimensions", embedding.length);

        // Get category classification from Python ML Service
        CategoryResponse categoryResponse = pythonMLService.classifyCategory(resumeFile);
        if (categoryResponse == null || !categoryResponse.getSuccess() || categoryResponse.getClassification() == null) {
            throw new IllegalStateException("Failed to classify resume category");
        }

        String category = categoryResponse.getClassification().getCategory();
        log.info("Resume classified as category: {}", category);

        // Call RPC to get similar jobs
        return rpcService.getSimilarJobsByCategory(embedding, category);
    }
}