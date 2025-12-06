package com.careermatch.backend.service;

import com.careermatch.backend.dto.response.*;
import com.careermatch.backend.repository.JobRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class JobServiceTest {

    @Mock
    private JobRepository jobRepository;

    @Mock
    private PythonMLService pythonMLService;

    @Mock
    private RpcService rpcService;

    @InjectMocks
    private JobService jobService;

    private MultipartFile mockResumeFile;
    private EmbeddingResponse mockEmbeddingResponse;
    private CategoryResponse mockCategoryResponse;
    private SimilarJobsResponse mockSimilarJobsResponse;

    @BeforeEach
    void setUp() {
        mockResumeFile = new MockMultipartFile(
                "file",
                "test-resume.pdf",
                "application/pdf",
                "Test resume content".getBytes());

        Double[] embedding = new Double[] { 0.1, 0.2, 0.3, 0.4, 0.5 };
        mockEmbeddingResponse = EmbeddingResponse.builder()
                .success(true)
                .embedding(embedding)
                .build();

        Classification classification = Classification.builder()
                .category("ENGINEERING")
                .confidence(0.95)
                .top3(Arrays.asList(
                        ClassificationDetail.builder().category("ENGINEERING").confidence(0.95).build(),
                        ClassificationDetail.builder().category("IT").confidence(0.03).build(),
                        ClassificationDetail.builder().category("DATA_SCIENCE").confidence(0.02).build()))
                .build();

        mockCategoryResponse = CategoryResponse.builder()
                .success(true)
                .classification(classification)
                .build();

        List<JobWithScore> jobs = Collections.singletonList(
                JobWithScore.builder()
                        .id("1")
                        .title("Software Engineer")
                        .company("Tech Corp")
                        .category("ENGINEERING")
                        .score(0.95)
                        .build());

        mockSimilarJobsResponse = SimilarJobsResponse.builder()
                .success(true)
                .jobs(jobs)
                .count(1)
                .message("Successfully found 1 similar jobs")
                .build();
    }

    @Test
    @DisplayName("Should get similar jobs with ML classification")
    void getSimilarJobsByResume_WithMLClassification_Success() {
        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(pythonMLService.classifyCategory(any(MultipartFile.class)))
                .thenReturn(mockCategoryResponse);
        when(rpcService.getSimilarJobsByCategory(any(), eq("ENGINEERING"), eq(10)))
                .thenReturn(mockSimilarJobsResponse);

        SimilarJobsResponse response = jobService.getSimilarJobsByResume(mockResumeFile, null, 10);

        assertNotNull(response);
        assertTrue(response.getSuccess());
        assertEquals("ENGINEERING", response.getCategory());
        assertEquals(1, response.getCount());
        assertNotNull(response.getCategories());
        assertEquals(3, response.getCategories().size());

        verify(pythonMLService).generateEmbedding(any(MultipartFile.class));
        verify(pythonMLService).classifyCategory(any(MultipartFile.class));
        verify(rpcService).getSimilarJobsByCategory(any(), eq("ENGINEERING"), eq(10));
    }

    @Test
    @DisplayName("Should get similar jobs with category override")
    void getSimilarJobsByResume_WithCategoryOverride_Success() {
        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(rpcService.getSimilarJobsByCategory(any(), eq("SALES"), eq(5)))
                .thenReturn(mockSimilarJobsResponse);

        SimilarJobsResponse response = jobService.getSimilarJobsByResume(mockResumeFile, "sales", 5);

        assertNotNull(response);
        assertTrue(response.getSuccess());
        assertEquals("SALES", response.getCategory());

        verify(pythonMLService).generateEmbedding(any(MultipartFile.class));
        verify(pythonMLService, never()).classifyCategory(any(MultipartFile.class));
        verify(rpcService).getSimilarJobsByCategory(any(), eq("SALES"), eq(5));
    }

    @Test
    @DisplayName("Should throw exception when embedding generation fails - null response")
    void getSimilarJobsByResume_EmbeddingFails_NullResponse() {
        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(null);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> jobService.getSimilarJobsByResume(mockResumeFile, null, 10));

        assertEquals("Failed to generate embeddings from resume", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when embedding generation fails - success false")
    void getSimilarJobsByResume_EmbeddingFails_SuccessFalse() {
        EmbeddingResponse failedResponse = EmbeddingResponse.builder()
                .success(false)
                .embedding(null)
                .build();

        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(failedResponse);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> jobService.getSimilarJobsByResume(mockResumeFile, null, 10));

        assertEquals("Failed to generate embeddings from resume", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when embedding is null")
    void getSimilarJobsByResume_EmbeddingFails_EmbeddingNull() {
        EmbeddingResponse failedResponse = EmbeddingResponse.builder()
                .success(true)
                .embedding(null)
                .build();

        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(failedResponse);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> jobService.getSimilarJobsByResume(mockResumeFile, null, 10));

        assertEquals("Failed to generate embeddings from resume", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when category classification fails - null response")
    void getSimilarJobsByResume_CategoryFails_NullResponse() {
        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(pythonMLService.classifyCategory(any(MultipartFile.class)))
                .thenReturn(null);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> jobService.getSimilarJobsByResume(mockResumeFile, null, 10));

        assertEquals("Failed to classify resume category", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when category classification fails - success false")
    void getSimilarJobsByResume_CategoryFails_SuccessFalse() {
        CategoryResponse failedResponse = CategoryResponse.builder()
                .success(false)
                .classification(null)
                .build();

        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(pythonMLService.classifyCategory(any(MultipartFile.class)))
                .thenReturn(failedResponse);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> jobService.getSimilarJobsByResume(mockResumeFile, null, 10));

        assertEquals("Failed to classify resume category", exception.getMessage());
    }

    @Test
    @DisplayName("Should throw exception when classification is null")
    void getSimilarJobsByResume_CategoryFails_ClassificationNull() {
        CategoryResponse failedResponse = CategoryResponse.builder()
                .success(true)
                .classification(null)
                .build();

        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(pythonMLService.classifyCategory(any(MultipartFile.class)))
                .thenReturn(failedResponse);

        IllegalStateException exception = assertThrows(IllegalStateException.class,
                () -> jobService.getSimilarJobsByResume(mockResumeFile, null, 10));

        assertEquals("Failed to classify resume category", exception.getMessage());
    }

    @Test
    @DisplayName("Should handle category with whitespace")
    void getSimilarJobsByResume_CategoryWithWhitespace() {
        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(rpcService.getSimilarJobsByCategory(any(), eq("ENGINEERING"), eq(10)))
                .thenReturn(mockSimilarJobsResponse);

        SimilarJobsResponse response = jobService.getSimilarJobsByResume(mockResumeFile, "  engineering  ", 10);

        assertNotNull(response);
        assertEquals("ENGINEERING", response.getCategory());
    }

    @Test
    @DisplayName("Should handle empty category override as no override")
    void getSimilarJobsByResume_EmptyCategoryOverride() {
        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(pythonMLService.classifyCategory(any(MultipartFile.class)))
                .thenReturn(mockCategoryResponse);
        when(rpcService.getSimilarJobsByCategory(any(), eq("ENGINEERING"), eq(10)))
                .thenReturn(mockSimilarJobsResponse);

        SimilarJobsResponse response = jobService.getSimilarJobsByResume(mockResumeFile, "", 10);

        verify(pythonMLService).classifyCategory(any(MultipartFile.class));
    }

    @Test
    @DisplayName("Should handle classification without top3")
    void getSimilarJobsByResume_ClassificationWithoutTop3() {
        Classification classificationNoTop3 = Classification.builder()
                .category("ENGINEERING")
                .confidence(0.95)
                .top3(null)
                .build();

        CategoryResponse categoryResponse = CategoryResponse.builder()
                .success(true)
                .classification(classificationNoTop3)
                .build();

        when(pythonMLService.generateEmbedding(any(MultipartFile.class)))
                .thenReturn(mockEmbeddingResponse);
        when(pythonMLService.classifyCategory(any(MultipartFile.class)))
                .thenReturn(categoryResponse);
        when(rpcService.getSimilarJobsByCategory(any(), eq("ENGINEERING"), eq(10)))
                .thenReturn(mockSimilarJobsResponse);

        SimilarJobsResponse response = jobService.getSimilarJobsByResume(mockResumeFile, null, 10);

        assertNotNull(response);
        assertEquals("ENGINEERING", response.getCategory());
        assertNull(response.getCategories());
    }
}
