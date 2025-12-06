package com.careermatch.backend.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SimilarJobsResponse {
    private Boolean success;
    private String category;
    private List<CategoryInfo> categories;
    private List<JobWithScore> jobs;
    private String message;
    private Integer count;

    @Data
    @Builder
    @NoArgsConstructor
    @AllArgsConstructor
    public static class CategoryInfo {
        private String category;
        private Integer confidence;
    }
}

