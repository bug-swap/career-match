package com.careermatch.backend.service;

import com.careermatch.backend.dto.response.*;
import com.careermatch.backend.exception.MLServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.IOException;

@Slf4j
@Service
@RequiredArgsConstructor
public class PythonMLService {

    private final WebClient pythonMLWebClient;

    /**
     * Calls Python API: POST /api/v1/resume/classify-sections
     */
    public SectionsResponse extractSections(MultipartFile file) {
        log.info("Calling Python ML API for sections: {}", file.getOriginalFilename());

        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            }).contentType(MediaType.APPLICATION_PDF);

            SectionsResponse response = pythonMLWebClient
                    .post()
                    .uri("/api/v1/resume/classify-sections")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
                            clientResponse -> clientResponse.bodyToMono(String.class)
                                    .flatMap(error -> Mono.error(new MLServiceException("Python API error: " + error))))
                    .bodyToMono(SectionsResponse.class)
                    .block();

            log.info("Python API returned sections successfully");
            return response;

        } catch (IOException e) {
            log.error("File read error: {}", e.getMessage());
            throw new MLServiceException("Failed to read file: " + e.getMessage());
        } catch (Exception e) {
            log.error("Python API call failed: {}", e.getMessage(), e);
            throw new MLServiceException("Python ML service unavailable at localhost:8000: " + e.getMessage());
        }
    }

    /**
     * Calls Python API: POST /api/v1/resume/extract-entities
     */
    public EntitiesResponse extractEntities(MultipartFile file) {
        log.info("Calling Python ML API for entities: {}", file.getOriginalFilename());

        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            }).contentType(MediaType.APPLICATION_PDF);

            EntitiesResponse response = pythonMLWebClient
                    .post()
                    .uri("/api/v1/resume/extract-entities")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
                            clientResponse -> clientResponse.bodyToMono(String.class)
                                    .flatMap(error -> Mono.error(new MLServiceException("Python API error: " + error))))
                    .bodyToMono(EntitiesResponse.class)
                    .block();

            log.info("Python API returned entities successfully");
            return response;

        } catch (IOException e) {
            log.error("File read error: {}", e.getMessage());
            throw new MLServiceException("Failed to read file: " + e.getMessage());
        } catch (Exception e) {
            log.error("Python API call failed: {}", e.getMessage(), e);
            throw new MLServiceException("Python ML service unavailable at localhost:8000: " + e.getMessage());
        }
    }

    /**
     * Calls Python API: POST /api/v1/resume/classify-category
     */
    public CategoryResponse classifyCategory(MultipartFile file) {
        log.info("Calling Python ML API for category: {}", file.getOriginalFilename());

        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            }).contentType(MediaType.APPLICATION_PDF);

            CategoryResponse response = pythonMLWebClient
                    .post()
                    .uri("/api/v1/resume/classify-category")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
                            clientResponse -> clientResponse.bodyToMono(String.class)
                                    .flatMap(error -> Mono.error(new MLServiceException("Python API error: " + error))))
                    .bodyToMono(CategoryResponse.class)
                    .block();

            log.info("Python API returned category successfully");
            return response;

        } catch (IOException e) {
            log.error("File read error: {}", e.getMessage());
            throw new MLServiceException("Failed to read file: " + e.getMessage());
        } catch (Exception e) {
            log.error("Python API call failed: {}", e.getMessage(), e);
            throw new MLServiceException("Python ML service unavailable at localhost:8000: " + e.getMessage());
        }
    }

    /**
    * Calls Python API: POST /api/v1/embedding
    */
    public EmbeddingResponse generateEmbedding(MultipartFile file) {
        log.info("Calling Python ML API for embedding: {}", file.getOriginalFilename());

        try {
            MultipartBodyBuilder builder = new MultipartBodyBuilder();
            builder.part("file", new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename();
                }
            }).contentType(MediaType.APPLICATION_PDF);

            EmbeddingResponse response = pythonMLWebClient
                    .post()
                    .uri("/api/v1/embedding")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(BodyInserters.fromMultipartData(builder.build()))
                    .retrieve()
                    .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
                            clientResponse -> clientResponse.bodyToMono(String.class)
                                    .flatMap(error -> Mono.error(new MLServiceException("Python API error: " + error))))
                    .bodyToMono(EmbeddingResponse.class)
                    .block();

            log.info("Python API returned embedding successfully");
            return response;

        } catch (IOException e) {
            log.error("File read error: {}", e.getMessage());
            throw new MLServiceException("Failed to read file: " + e.getMessage());
        } catch (Exception e) {
            log.error("Python API call failed: {}", e.getMessage(), e);
            throw new MLServiceException("Python ML service unavailable at localhost:8000: " + e.getMessage());
        }
    }
}