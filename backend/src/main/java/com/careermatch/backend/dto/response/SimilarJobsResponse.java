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
    private List<JobWithScore> jobs;
    private String message;
    private Integer count;
}

