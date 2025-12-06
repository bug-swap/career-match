package com.careermatch.backend.service;

import com.careermatch.backend.dto.response.JobWithScore;
import com.careermatch.backend.dto.response.SimilarJobsResponse;
import com.careermatch.backend.exception.MLServiceException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class RpcService {

    private final WebClient rpcWebClient;

    @Value("${supabase.anon.key:}")
    private String supabaseAnonKey;

    /**
     * Calls Supabase RPC function: get_similar_jobs_by_category
     * @param embedding The resume embedding vector
     * @param category The resume category
     * @return Similar jobs with similarity scores from Supabase RPC function
     */
    public SimilarJobsResponse getSimilarJobsByCategory(Double[] embedding, String category) {
        log.info("Calling Supabase RPC function get_similar_jobs_by_category for category: {}", category);

        try {
            // Prepare RPC request payload for Supabase PostgREST
            Map<String, Object> params = new HashMap<>();
            params.put("p_embedding", embedding);
            params.put("p_category", category);

            // Call Supabase RPC endpoint using PostgREST format
            List<JobWithScore> jobs = rpcWebClient
                    .post()
                    .uri("/rpc/get_similar_jobs_by_category")
                    .header("apikey", supabaseAnonKey)
                    .header("Authorization", "Bearer " + supabaseAnonKey)
                    .bodyValue(params)
                    .retrieve()
                    .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(),
                            clientResponse -> clientResponse.bodyToMono(String.class)
                                    .flatMap(error -> {
                                        log.error("Supabase RPC error response: {}", error);
                                        return Mono.error(new MLServiceException("Supabase RPC error: " + error));
                                    }))
                    .bodyToMono(new ParameterizedTypeReference<List<JobWithScore>>() {})
                    .block();

            int jobCount = jobs != null ? jobs.size() : 0;
            SimilarJobsResponse response = SimilarJobsResponse.builder()
                    .success(true)
                    .jobs(jobs)
                    .count(jobCount)
                    .message("Successfully found " + jobCount + " similar jobs")
                    .build();

            log.info("Supabase RPC call successful, found {} similar jobs", jobCount);
            return response;

        } catch (Exception e) {
            log.error("Supabase RPC call failed: {}", e.getMessage(), e);
            throw new MLServiceException("Supabase RPC service unavailable: " + e.getMessage());
        }
    }
}
